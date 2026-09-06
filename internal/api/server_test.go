package api

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestRankOrdersCandidatesByValueAndRelevance(t *testing.T) {
	server := (&Server{}).Handler()
	input := RankRequest{
		RequestID:      "request-1",
		QueryEmbedding: []float64{1, 0},
		Limit:          2,
		Candidates: []Candidate{
			{AdID: "lower", Embedding: []float64{1, 0}, PCTR: 0.1, BidMicros: 100, PacingMultiplier: 1},
			{AdID: "higher", Embedding: []float64{1, 0}, PCTR: 0.2, BidMicros: 100, PacingMultiplier: 1},
		},
	}
	body, _ := json.Marshal(input)
	request := httptest.NewRequest(http.MethodPost, "/v1/rank", bytes.NewReader(body))
	response := httptest.NewRecorder()
	server.ServeHTTP(response, request)

	if response.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", response.Code, response.Body.String())
	}
	var output RankResponse
	if err := json.Unmarshal(response.Body.Bytes(), &output); err != nil {
		t.Fatal(err)
	}
	if output.Candidates[0].AdID != "higher" {
		t.Fatalf("expected higher first, got %s", output.Candidates[0].AdID)
	}
}

func TestRankRejectsMismatchedDimensions(t *testing.T) {
	server := (&Server{}).Handler()
	body := []byte(`{"query_embedding":[1,0],"limit":1,"candidates":[{"ad_id":"a","embedding":[1],"pctr":0.1,"bid_micros":3,"pacing_multiplier":1}]}`)
	request := httptest.NewRequest(http.MethodPost, "/v1/rank", bytes.NewReader(body))
	response := httptest.NewRecorder()
	server.ServeHTTP(response, request)
	if response.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", response.Code)
	}
}
