package cache

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"strconv"
	"strings"
	"time"
)

type Store interface {
	Get(ctx context.Context, key string) ([]byte, bool, error)
	Set(ctx context.Context, key string, value []byte, ttl time.Duration) error
}

// Redis implements the small RESP2 subset needed by the rank-response cache.
// It uses one short-lived connection per operation to keep failure behavior
// simple and explicit in the reference service.
type Redis struct {
	Address string
	Timeout time.Duration
}

func (r *Redis) Get(ctx context.Context, key string) ([]byte, bool, error) {
	reader, conn, err := r.command(ctx, "GET", key)
	if err != nil {
		return nil, false, err
	}
	defer conn.Close()

	prefix, err := reader.ReadString('\n')
	if err != nil {
		return nil, false, err
	}
	if prefix == "$-1\r\n" {
		return nil, false, nil
	}
	if !strings.HasPrefix(prefix, "$") {
		return nil, false, fmt.Errorf("unexpected Redis GET response %q", prefix)
	}
	length, err := strconv.Atoi(strings.TrimSpace(strings.TrimPrefix(prefix, "$")))
	if err != nil || length < 0 {
		return nil, false, fmt.Errorf("invalid Redis bulk length %q", prefix)
	}
	payload := make([]byte, length+2)
	if _, err := io.ReadFull(reader, payload); err != nil {
		return nil, false, err
	}
	return payload[:length], true, nil
}

func (r *Redis) Set(ctx context.Context, key string, value []byte, ttl time.Duration) error {
	seconds := max(int(ttl/time.Second), 1)
	reader, conn, err := r.command(ctx, "SETEX", key, strconv.Itoa(seconds), string(value))
	if err != nil {
		return err
	}
	defer conn.Close()
	response, err := reader.ReadString('\n')
	if err != nil {
		return err
	}
	if response != "+OK\r\n" {
		return fmt.Errorf("unexpected Redis SETEX response %q", response)
	}
	return nil
}

func (r *Redis) command(ctx context.Context, arguments ...string) (*bufio.Reader, net.Conn, error) {
	timeout := r.Timeout
	if timeout <= 0 {
		timeout = 100 * time.Millisecond
	}
	dialer := net.Dialer{Timeout: timeout}
	conn, err := dialer.DialContext(ctx, "tcp", r.Address)
	if err != nil {
		return nil, nil, err
	}
	deadline := time.Now().Add(timeout)
	if contextDeadline, ok := ctx.Deadline(); ok && contextDeadline.Before(deadline) {
		deadline = contextDeadline
	}
	if err := conn.SetDeadline(deadline); err != nil {
		conn.Close()
		return nil, nil, err
	}

	var command strings.Builder
	fmt.Fprintf(&command, "*%d\r\n", len(arguments))
	for _, argument := range arguments {
		fmt.Fprintf(&command, "$%d\r\n%s\r\n", len(argument), argument)
	}
	if _, err := io.WriteString(conn, command.String()); err != nil {
		conn.Close()
		return nil, nil, err
	}
	return bufio.NewReader(conn), conn, nil
}

var ErrCacheMiss = errors.New("cache miss")
