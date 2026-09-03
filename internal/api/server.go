package api

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"math"
	"net/http"
	"sort"
	"time"

	"github.com/AbhinavShukla2020/ads-ranking-pacing/internal/cache"
)

type Candidate struct {
	AdID             string    `json:"ad_id"`
	Embedding        []float64 `json:"embedding"`
	PCTR             float64   `json:"pctr"`
	BidMicros        int64     `json:"bid_micros"`
	PacingMultiplier float64   `json:"pacing_multiplier"`
}

type RankRequest struct {
	RequestID      string      `json:"request_id"`
	QueryEmbedding []float64   `json:"query_embedding"`
	Candidates     []Candidate `json:"candidates"`
	Limit          int         `json:"limit"`
}

type RankedCandidate struct {
	AdID           string  `json:"ad_id"`
	RetrievalScore float64 `json:"retrieval_score"`
	ValueScore     float64 `json:"value_score"`
	RankScore      float64 `json:"rank_score"`
}

type RankResponse struct {
	RequestID  string            `json:"request_id"`
	Candidates []RankedCandidate `json:"candidates"`
	Cached     bool              `json:"cached"`
}

type Server struct {
	Cache         cache.Store
	CacheTTL      time.Duration
	MaxBodyBytes  int64
	MaxCandidates int
	MaxDimensions int
	Logger        *slog.Logger
}

func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", s.health)
	mux.HandleFunc("POST /v1/rank", s.rank)
	return mux
}

func (s *Server) health(writer http.ResponseWriter, _ *http.Request) {
	writeJSON(writer, http.StatusOK, map[string]string{"status": "ok"})
}

func (s *Server) rank(writer http.ResponseWriter, request *http.Request) {
	maximumBody := s.MaxBodyBytes
	if maximumBody <= 0 {
		maximumBody = 2 << 20
	}
	body, err := io.ReadAll(http.MaxBytesReader(writer, request.Body, maximumBody))
	if err != nil {
		writeError(writer, http.StatusRequestEntityTooLarge, "request body is too large")
		return
	}
	cacheKey := bodyCacheKey(body)
	if payload, found := s.cacheGet(request.Context(), cacheKey); found {
		writer.Header().Set("Content-Type", "application/json")
		writer.Header().Set("X-Cache", "HIT")
		writer.WriteHeader(http.StatusOK)
		_, _ = writer.Write(markCached(payload))
		return
	}

	var input RankRequest
	decoder := json.NewDecoder(bytesReader(body))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&input); err != nil {
		writeError(writer, http.StatusBadRequest, "invalid JSON request")
		return
	}
	if err := s.validate(input); err != nil {
		writeError(writer, http.StatusBadRequest, err.Error())
		return
	}

	ranked := make([]RankedCandidate, 0, len(input.Candidates))
	for _, candidate := range input.Candidates {
		relevance := cosine(input.QueryEmbedding, candidate.Embedding)
		value := candidate.PCTR * float64(candidate.BidMicros) * candidate.PacingMultiplier
		rankScore := value * math.Max(0, relevance)
		ranked = append(ranked, RankedCandidate{
			AdID:           candidate.AdID,
			RetrievalScore: relevance,
			ValueScore:     value,
			RankScore:      rankScore,
		})
	}
	sort.Slice(ranked, func(i, j int) bool {
		if ranked[i].RankScore == ranked[j].RankScore {
			return ranked[i].AdID < ranked[j].AdID
		}
		return ranked[i].RankScore > ranked[j].RankScore
	})
	if input.Limit < len(ranked) {
		ranked = ranked[:input.Limit]
	}
	response := RankResponse{RequestID: input.RequestID, Candidates: ranked, Cached: false}
	payload, err := json.Marshal(response)
	if err != nil {
		writeError(writer, http.StatusInternalServerError, "could not encode response")
		return
	}
	s.cacheSet(request.Context(), cacheKey, payload)
	writer.Header().Set("X-Cache", "MISS")
	writeRawJSON(writer, http.StatusOK, payload)
}

func (s *Server) validate(input RankRequest) error {
	maximumCandidates := s.MaxCandidates
	if maximumCandidates <= 0 {
		maximumCandidates = 2_000
	}
	maximumDimensions := s.MaxDimensions
	if maximumDimensions <= 0 {
		maximumDimensions = 512
	}
	if len(input.QueryEmbedding) == 0 || len(input.QueryEmbedding) > maximumDimensions {
		return errors.New("query_embedding has an invalid dimension")
	}
	if len(input.Candidates) == 0 || len(input.Candidates) > maximumCandidates {
		return errors.New("candidates must be non-empty and within the configured limit")
	}
	if input.Limit <= 0 || input.Limit > len(input.Candidates) {
		return errors.New("limit must be between 1 and the candidate count")
	}
	for _, candidate := range input.Candidates {
		if candidate.AdID == "" || len(candidate.Embedding) != len(input.QueryEmbedding) {
			return errors.New("each candidate needs an id and a matching embedding dimension")
		}
		if candidate.PCTR < 0 || candidate.PCTR > 1 || candidate.BidMicros < 0 || candidate.PacingMultiplier < 0 {
			return errors.New("candidate probabilities, bids, and multipliers are out of range")
		}
	}
	return nil
}

func cosine(left, right []float64) float64 {
	var dot, leftNorm, rightNorm float64
	for index := range left {
		dot += left[index] * right[index]
		leftNorm += left[index] * left[index]
		rightNorm += right[index] * right[index]
	}
	if leftNorm == 0 || rightNorm == 0 {
		return 0
	}
	return dot / math.Sqrt(leftNorm*rightNorm)
}

func bodyCacheKey(body []byte) string {
	digest := sha256.Sum256(body)
	return "rank:" + hex.EncodeToString(digest[:])
}

func (s *Server) cacheGet(ctx context.Context, key string) ([]byte, bool) {
	if s.Cache == nil {
		return nil, false
	}
	payload, found, err := s.Cache.Get(ctx, key)
	if err != nil && s.Logger != nil {
		s.Logger.Warn("cache read failed", "error", err)
	}
	return payload, found && err == nil
}

func (s *Server) cacheSet(ctx context.Context, key string, payload []byte) {
	if s.Cache == nil || s.CacheTTL <= 0 {
		return
	}
	if err := s.Cache.Set(ctx, key, payload, s.CacheTTL); err != nil && s.Logger != nil {
		s.Logger.Warn("cache write failed", "error", err)
	}
}

func markCached(payload []byte) []byte {
	var response RankResponse
	if json.Unmarshal(payload, &response) != nil {
		return payload
	}
	response.Cached = true
	updated, err := json.Marshal(response)
	if err != nil {
		return payload
	}
	return updated
}

func writeError(writer http.ResponseWriter, status int, message string) {
	writeJSON(writer, status, map[string]string{"error": message})
}

func writeJSON(writer http.ResponseWriter, status int, value any) {
	payload, err := json.Marshal(value)
	if err != nil {
		http.Error(writer, "could not encode response", http.StatusInternalServerError)
		return
	}
	writeRawJSON(writer, status, payload)
}

func writeRawJSON(writer http.ResponseWriter, status int, payload []byte) {
	writer.Header().Set("Content-Type", "application/json")
	writer.WriteHeader(status)
	_, _ = writer.Write(append(payload, '\n'))
}

type byteReader struct {
	payload []byte
	offset  int
}

func bytesReader(payload []byte) *byteReader {
	return &byteReader{payload: payload}
}

func (reader *byteReader) Read(destination []byte) (int, error) {
	if reader.offset >= len(reader.payload) {
		return 0, io.EOF
	}
	count := copy(destination, reader.payload[reader.offset:])
	reader.offset += count
	return count, nil
}
