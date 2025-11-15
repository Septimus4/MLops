# Drift Monitoring

## Overview

Feature drift occurs when the statistical properties of input features change over time. This can indicate:
- Changes in the underlying population
- Data quality issues
- Upstream system changes
- Seasonal variations
- Model becoming outdated

Monitoring drift is crucial for maintaining model performance in production.

## How It Works

### 1. Baseline Statistics

During training, we compute baseline statistics for each feature:
- Mean (μ)
- Standard deviation (σ)
- Min, max, median

These are saved to `data/artifacts/baseline_stats.json`.

### 2. Live Statistics

For each time window:
- Fetch predictions from the database
- Extract feature values
- Compute live mean for each feature

### 3. Drift Computation

For each feature, we compute the z-score:

```
z = |μ_live - μ_train| / σ_train
```

Where:
- `μ_live` = mean of feature in live data
- `μ_train` = mean of feature in training data
- `σ_train` = standard deviation in training data

### 4. Interpretation

Z-scores indicate how many standard deviations the live mean differs from the training mean:

- **z < 1.0**: No significant drift
  - Live data similar to training data
  - No action needed

- **1.0 ≤ z < 2.0**: Moderate drift
  - Notable change in feature distribution
  - Monitor closely
  - Investigate if multiple features affected

- **z ≥ 2.0**: Significant drift
  - Major change in feature distribution
  - Investigate immediately
  - Consider model retraining

## Using the Streamlit Dashboard

### Accessing the Dashboard

```bash
# Local
streamlit run src/ui/streamlit_drift.py

# Docker
docker-compose up streamlit
```

Navigate to http://localhost:8501

### Dashboard Features

#### Time Window Selection

Choose the time period for drift analysis:
- 1 hour: Recent drift, high sensitivity
- 6 hours: Short-term trends
- 24 hours: Daily patterns (default)
- 48 hours: Multi-day trends
- 72 hours: Weekly patterns
- 168 hours (1 week): Long-term drift

#### Metrics Display

**Summary Cards**:
- Time window
- Number of predictions analyzed
- Maximum drift (highest z-score)

**Drift Table**:
- All features with computed drift
- Color-coded by severity
- Sortable columns

**Top 10 Chart**:
- Bar chart of most drifting features
- Visual comparison of drift magnitudes

**Detailed View**:
- Top 5 drifting features expanded
- Training vs. live means
- Percentage change calculation

#### Auto-Refresh

Enable auto-refresh to monitor drift in real-time:
1. Check "Auto-refresh" in sidebar
2. Set refresh interval (10-300 seconds)
3. Dashboard updates automatically

## Using the API

### Query Drift Endpoint

```bash
curl "http://localhost:8000/drift?window_hours=24"
```

### Python Example

```python
import requests
import pandas as pd

# Get drift metrics
response = requests.get(
    "http://localhost:8000/drift",
    params={"window_hours": 24}
)

data = response.json()

# Convert to DataFrame
df = pd.DataFrame(data['metrics'])

# Filter for significant drift
significant = df[df['z_score'] >= 2.0]

print(f"Found {len(significant)} features with significant drift:")
for _, row in significant.iterrows():
    print(f"  {row['feature_name']}: z={row['z_score']:.2f}")
```

### Scheduled Monitoring

Set up automated drift monitoring:

```python
import schedule
import time
import requests

def check_drift():
    response = requests.get(
        "http://localhost:8000/drift",
        params={"window_hours": 24}
    )
    
    data = response.json()
    max_z = max(m['z_score'] for m in data['metrics']) if data['metrics'] else 0
    
    if max_z >= 2.0:
        print(f"ALERT: Significant drift detected (z={max_z:.2f})")
        # Send alert (email, Slack, PagerDuty, etc.)
    else:
        print(f"Drift check OK (max z={max_z:.2f})")

# Check every hour
schedule.every().hour.do(check_drift)

while True:
    schedule.run_pending()
    time.sleep(60)
```

## Common Drift Patterns

### 1. Gradual Drift

**Characteristics**:
- Z-scores slowly increase over days/weeks
- Multiple features affected
- Smooth transition

**Causes**:
- Population changes
- Economic trends
- Seasonal effects

**Response**:
- Monitor trend
- Plan model retraining
- Investigate root cause

### 2. Sudden Drift

**Characteristics**:
- Abrupt z-score increase
- Single or few features affected
- Sharp transition

**Causes**:
- Data pipeline bug
- Upstream system change
- Data quality issue

**Response**:
- Investigate immediately
- Check data sources
- Rollback if needed
- Fix pipeline

### 3. Cyclical Drift

**Characteristics**:
- Z-scores vary periodically
- Predictable pattern
- Returns to baseline

**Causes**:
- Daily/weekly cycles
- Seasonal patterns
- Business cycles

**Response**:
- Document pattern
- Adjust monitoring thresholds
- Consider time-aware features

### 4. No Drift

**Characteristics**:
- Z-scores remain low (< 1.0)
- Stable over time
- Similar to training data

**Interpretation**:
- Model inputs stable
- Production data matches training
- Good data pipeline health

## Best Practices

### 1. Establish Baselines

- Compute baseline stats from representative training data
- Update baselines when retraining
- Version baseline files with models

### 2. Set Appropriate Windows

- **Short windows (1-6 hours)**: Detect immediate issues
- **Medium windows (24-48 hours)**: Daily monitoring
- **Long windows (1 week)**: Trend analysis

### 3. Define Thresholds

Customize z-score thresholds based on:
- Feature importance
- Business tolerance
- Historical patterns

### 4. Automate Alerts

Set up automated alerts for:
- z > 2.0: Immediate notification
- z > 1.5 for 3+ consecutive checks: Warning
- No predictions for 1 hour: System check

### 5. Correlate with Business Events

Track:
- Product launches
- Marketing campaigns
- Policy changes
- External events

### 6. Log Everything

Maintain records of:
- Drift detection events
- Investigations
- Actions taken
- Outcomes

## Limitations

### Current Implementation

1. **Univariate Analysis**: Each feature analyzed independently
   - Doesn't detect multivariate drift
   - Misses correlation changes

2. **Z-Score Only**: Single drift metric
   - Doesn't detect distribution shape changes
   - Assumes normal distribution

3. **Mean-Based**: Only tracks mean changes
   - Doesn't detect variance changes
   - Misses tail distribution changes

4. **No Automatic Response**: Manual investigation required
   - No auto-retraining
   - No auto-alerting

### Future Enhancements

Consider adding:

1. **Multivariate Tests**:
   - Hotelling's T² test
   - Mahalanobis distance

2. **Distribution Tests**:
   - Kolmogorov-Smirnov test
   - Chi-square test
   - Population Stability Index (PSI)

3. **Performance Monitoring**:
   - Prediction distribution drift
   - Model confidence drift
   - Actual outcome tracking (if available)

4. **Advanced Alerting**:
   - Email/Slack notifications
   - PagerDuty integration
   - Configurable thresholds per feature

5. **Automatic Retraining**:
   - Trigger retraining on drift
   - A/B test new models
   - Gradual rollout

## Troubleshooting

### No Drift Metrics

**Problem**: Drift endpoint returns empty metrics

**Solutions**:
- Ensure predictions are being logged
- Check database connection
- Verify time window includes predictions
- Confirm baseline stats are loaded

### Incorrect Drift Values

**Problem**: Z-scores don't match expectations

**Solutions**:
- Verify baseline stats match current model
- Check feature preprocessing consistency
- Ensure timestamp handling is correct
- Validate database query logic

### Dashboard Not Updating

**Problem**: Streamlit dashboard shows stale data

**Solutions**:
- Click refresh button
- Check API connectivity
- Verify API is running
- Check browser console for errors

### High Drift on All Features

**Problem**: All features show significant drift

**Solutions**:
- Check if baseline stats are correct
- Verify model is loaded properly
- Investigate data pipeline
- Check for preprocessing bugs

## References

- [Concept Drift in Machine Learning](https://en.wikipedia.org/wiki/Concept_drift)
- [Monitoring Machine Learning Models in Production](https://christophergs.com/machine%20learning/2020/03/14/how-to-monitor-machine-learning-models/)
- [Data Distribution Shifts and Monitoring](https://huyenchip.com/2022/02/07/data-distribution-shifts-and-monitoring.html)
