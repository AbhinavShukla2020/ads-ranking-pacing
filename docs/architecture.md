# Architecture notes

## Offline path

The synthetic generator produces user, ad, context, click, conversion, bid, and
price fields from a latent-factor process. Positives are therefore learnable
without making the sample unrealistically separable.

The retrieval model embeds user/context and ad features into the same normalized
space. Training uses in-batch negatives: the diagonal of the similarity matrix
is the observed pair, while other ads in the mini-batch become negatives. At
index time, ad vectors are added to a FAISS inner-product index. The NumPy index
implements the same interface for machines where FAISS is unavailable.

The ranker concatenates dense user, ad, and context features. Cross layers model
bounded-degree feature interactions, while a deep branch learns less structured
interactions. Two heads estimate click probability and conversion probability
conditional on click. Their product is the impression-level conversion
probability used by the ESMM loss.

## Auction path

Each eligible candidate has a base bid and a pacing multiplier. The effective
bid enters a deterministic second-price auction. The winner pays the larger of
the reserve and runner-up bid, capped at its own effective bid. Stable ad IDs
break exact ties, which makes replays reproducible.

The pacing controller compares cumulative spend with an ideal spend curve. Its
proportional term reacts immediately, the integral term corrects persistent
under-delivery, and the derivative term damps sudden changes. Integral clamping
prevents a long period with no inventory from causing an extreme correction.

## Online boundary

The Go service validates vector dimensions and candidate counts before scoring.
It computes inner-product relevance and expected-value scores, then returns the
top candidates. Responses can be cached in Redis for a short TTL using a request
key supplied by the caller. Redis failures are treated as cache misses so the
ranking path remains available.

The reference Redis client implements the small RESP subset the server needs,
keeping the Go binary dependency-free. A production service should use a mature
client with connection pooling, telemetry, authentication, and TLS.
