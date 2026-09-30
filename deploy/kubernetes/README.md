# Kubernetes

SignalRank-RAG includes a small Kubernetes runtime for local validation and deployment testing.

It provides:

- Deployment and ClusterIP Service
- startup, readiness, and liveness probes
- CPU and memory requests/limits
- HorizontalPodAutoscaler
- Secret-based service-token injection
- ConfigMap-based smoke configuration
- init-container data seeding
- Kustomize composition

The smoke runtime uses a small embedding model, local Qdrant storage, and no LLM provider, so it can run without external AI credentials.

## Run locally

```bash
kind create cluster --name signalrank

kubectl apply -f deploy/kubernetes/base/namespace.yaml

kubectl create secret generic signalrank-secrets \
  -n signalrank \
  --from-literal=SIGNALRANK_SERVICE_TOKEN=local-dev-token

kubectl apply -k deploy/kubernetes

kubectl rollout status deployment/signalrank-api \
  -n signalrank \
  --timeout=420s
````

Forward the Service:

```bash
kubectl port-forward \
  -n signalrank \
  service/signalrank-api \
  18000:8000
```

Health check:

```bash
curl http://127.0.0.1:18000/health
```

Authenticated retrieval:

```bash
curl -X POST http://127.0.0.1:18000/retrieve \
  -H "Content-Type: application/json" \
  -H "X-SignalRank-Service-Token: local-dev-token" \
  -d '{"query":"Mars rover","mode":"bm25","top_k":2}'
```

## Troubleshooting

```bash
kubectl get pods -n signalrank
kubectl logs -n signalrank deployment/signalrank-api
kubectl describe pod -n signalrank <pod-name>
kubectl get events -n signalrank --sort-by=.lastTimestamp
kubectl apply --dry-run=server -k deploy/kubernetes
```

The default local `kind` cluster has no metrics-server, so HPA CPU metrics may appear as `<unknown>`.

## Cleanup

```bash
kind delete cluster --name signalrank
```

The hosted SignalRank-RAG deployment remains on Azure Container Apps. Kubernetes is an additional validated runtime target.