.PHONY: test lint train simulate serve

test:
	python3 -m pytest

lint:
	python3 -m ruff check src tests benchmarks

train:
	python3 -m ads_engine.train --events 100000 --epochs 3 --output artifacts

simulate:
	python3 benchmarks/auction_sim.py --auctions 50000 --seed 7

serve:
	go run ./cmd/server
