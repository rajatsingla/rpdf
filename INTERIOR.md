# Interior PDF fix

```
POST /rpdf/interiors
Content-Type: application/pdf

<raw PDF bytes>
```

Returns `200` with the fixed PDF as raw bytes (`application/pdf`), or `400` on
an empty body or a PDF that could not be processed.

```sh
curl -X POST 'http://localhost:8000/rpdf/interiors' \
  --data-binary @interior.pdf \
  -o interior-fixed.pdf
```
