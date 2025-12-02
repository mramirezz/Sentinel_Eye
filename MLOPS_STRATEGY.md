# MLOps Strategy: Sentinel Eye at Scale

## Architecture

**Edge (Jetson Xavier NX)**: Docker + K3s, 24h buffer, Prometheus exporter  
**Regional Hub**: TimescaleDB (90d), Grafana, MinIO  
**Cloud (AWS)**: Kinesis, S3, Redshift, SageMaker, MLflow

## Data Drift Detection

**Baseline**: 30 días iniciales por cámara (mean, std, p95)

**Detection**: Diaria via Kolmogorov-Smirnov test (p < 0.05)

**SQL Query**:
```sql
SELECT camera_id, (baseline_mean - recent_mean) as drift
FROM baseline JOIN recent USING(camera_id)
WHERE drift > 5;
```

**Auto-Remediation**:
- QC < 40 (24h) → Auto ticket "limpiar cámara"
- Vibration > 5px (6h) → Ticket "revisar montaje"  
- Drift 3 días → Trigger retraining

## CI/CD Pipeline

```yaml
test → build → push ECR → deploy canary (10%) → monitor 1h → promote/rollback
```

## Monitoring

**Metrics**: `qc_score`, `vibration_magnitude`, `frame_processing_ms`, `detections_total`

**Alerts**:
- QC < 40 (30m) → Warning
- Vibration > 5px (1h) → Critical
- Latency p95 > 100ms (10m) → Warning

**Dashboard**: Grafana con fleet overview, QC heatmap, latency p95

## Model Retraining

1. Collect edge cases (QC < 60, vibration > 3px)
2. SageMaker training (80% historical + 20% edge cases)
3. A/B test: 10% dispositivos, 24h
4. Promote si t-test p < 0.05 y mejora QC

## Cost (5 years, 1000 cameras)

- Edge hardware: $1.1M
- Edge operations: $270k/year
- Cloud (AWS): $66k/year
- **Total**: $2.78M

---

**Version**: 2.1 | **Author**: mramirezz | **Date**: Dec 2, 2025
