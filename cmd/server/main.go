package main

import (
	"log/slog"
	"net/http"
	"os"
	"strconv"
	"time"

	"github.com/AbhinavShukla2020/ads-ranking-pacing/internal/api"
	"github.com/AbhinavShukla2020/ads-ranking-pacing/internal/cache"
)

func main() {
	logger := slog.New(slog.NewJSONHandler(os.Stdout, nil))
	address := environment("HTTP_ADDR", ":8080")
	ttlSeconds := environmentInteger("CACHE_TTL_SECONDS", 5)

	server := &api.Server{
		CacheTTL:      time.Duration(ttlSeconds) * time.Second,
		MaxBodyBytes:  2 << 20,
		MaxCandidates: 2_000,
		MaxDimensions: 512,
		Logger:        logger,
	}
	if redisAddress := os.Getenv("REDIS_ADDR"); redisAddress != "" {
		server.Cache = &cache.Redis{Address: redisAddress, Timeout: 100 * time.Millisecond}
	}

	httpServer := &http.Server{
		Addr:              address,
		Handler:           server.Handler(),
		ReadHeaderTimeout: 2 * time.Second,
		ReadTimeout:       3 * time.Second,
		WriteTimeout:      3 * time.Second,
		IdleTimeout:       30 * time.Second,
	}
	logger.Info("ads ranking server listening", "address", address)
	if err := httpServer.ListenAndServe(); err != nil {
		logger.Error("server stopped", "error", err)
		os.Exit(1)
	}
}

func environment(name, fallback string) string {
	if value := os.Getenv(name); value != "" {
		return value
	}
	return fallback
}

func environmentInteger(name string, fallback int) int {
	value, err := strconv.Atoi(os.Getenv(name))
	if err != nil || value <= 0 {
		return fallback
	}
	return value
}
