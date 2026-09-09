# API Rate Limits

The Nimbus REST API enforces the following default rate limits per API key:
Starter: 60 requests/minute. Team: 300 requests/minute. Business: 1200
requests/minute. Enterprise: custom limits negotiated per contract, typically
starting at 5000 requests/minute. Requests exceeding the limit receive an HTTP
429 response with a Retry-After header. Sustained abuse beyond 10x the limit may
result in temporary API key suspension.
