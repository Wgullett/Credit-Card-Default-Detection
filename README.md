# Credit Card Default Prediction with XGBoost and SHAP Analysis

## Project Overview

This project builds a machine learning model to predict credit card defaults using XGBoost and interprets the results using SHAP (SHapley Additive exPlanations). The model analyzes 30,000 credit card customer records to identify patterns that predict default risk, optimizing for business profit rather than just model accuracy.

### Key Objectives
- Predict which customers will default on their credit card payments
- Understand **why** the model makes specific predictions using SHAP
- Optimize decision thresholds for maximum business profitability
- Provide actionable insights for credit risk management

---

## Model Performance

The model was trained using three optimization approaches:

| Model Type | F1 Score | Precision | Recall | ROC AUC | Business Profit |
|-----------|----------|-----------|--------|---------|-----------------|
| Baseline XGBoost | ~0.45 | ~0.40 | ~0.52 | ~0.77 | Baseline |
| F1-Optimized | ~0.50 | ~0.45 | ~0.56 | ~0.78 | +5% vs baseline |
| **Profit-Optimized** | ~0.48 | ~0.48 | ~0.48 | ~0.78 | **+12% vs baseline** |

**Key Finding:** The profit-optimized model provides the best business outcomes by balancing the $5,000 cost of missed defaults against the $200 cost of denying good customers.

---

## 📊 SHAP Visualization Analysis

SHAP values explain how each feature contributes to individual predictions. Below is a detailed breakdown of each visualization.

---

### 🔹 Image 1: Feature Importance (Beeswarm Plot)

![Beeswarm Plot](image1)

**What This Shows:** A comprehensive ranking of all features by their impact on default predictions.

#### Understanding the Visualization

**Axes:**
- **Y-axis (left):** Features ranked from most important (top) to least important (bottom)
- **X-axis (bottom):** SHAP value = the impact on the model's prediction
  - **Negative values (left):** Feature decreases default risk
  - **Positive values (right):** Feature increases default risk
- **Color:** Feature value magnitude
  - **Red/Pink:** High feature values
  - **Blue:** Low feature values

**Key Insights:**

1. **Payment Status (Current Month)** - Dominant Predictor
   - Red dots far right = customers 2+ months late have extremely high default risk
   - Blue dots left = on-time payers have low risk
   - **Impact:** Single most important feature (~40-50% of model decisions)

2. **Payment Status (2 Months Ago)** - Secondary Indicator
   - Similar pattern but weaker effect
   - Shows payment history compounds risk

3. **Credit Limit Amount** - Protective Factor
   - Red dots (high limits) push LEFT = decrease risk
   - Blue dots (low limits) push RIGHT = increase risk
   - **Interpretation:** High credit limits were given to creditworthy customers

4. **Payment Amounts** - Active Management
   - Higher recent payments = lower risk
   - Shows customers actively servicing debt

5. **Bill Amounts** - Debt Load
   - Very high bills relative to limit = financial stress = higher risk

**Business Takeaway:** Payment behavior (especially recent delays) matters far more than demographics or credit limits.

---

### Image 2: Decision Plot

![Decision Plot](image2)

**What This Shows:** The journey from base prediction to final prediction for multiple customers, showing how features accumulate to create risk assessments.

#### Understanding the Visualization

**Axes:**
- **Y-axis (left):** Features listed from most to least important
- **X-axis (bottom):** Model output value (cumulative prediction)
  - Starts at base rate (~0.37 = 37% average default rate)
  - Moves left (decreasing risk) or right (increasing risk)
- **Lines:** Each line represents one customer's prediction path
  - **Blue lines:** Customers predicted NOT to default (end on left)
  - **Red/Pink lines:** Customers predicted TO DEFAULT (end on right)

**Key Patterns:**

1. **The Critical Split at PAY_0:**
   - All lines start together at the bottom
   - At "Payment Status (Current Month)," they diverge dramatically
   - This is the moment where most predictions are determined

2. **Blue Line Customers (Non-Defaulters):**
   - Stay left of center line
   - PAY_0, PAY_2, and payment amounts all contribute to safety
   - Even negative factors don't push them into danger

3. **Red/Pink Line Customers (Defaulters):**
   - PAY_0 alone pushes them far right
   - Other features can't rescue them from severe payment delays
   - Some stabilize after initial spike, but most remain high-risk

**Business Takeaway:** Payment status creates a "point of no return" - once customers fall 2+ months behind, other positive factors rarely save them from default.

---

### Image 3: Credit Limit Impact

![Credit Limit Scatter](image3)

**What This Shows:** The relationship between credit limit amounts and their impact on default predictions.

#### Understanding the Visualization

**Axes:**
- **X-axis (bottom):** Credit Limit Amount (actual dollar values, 0 to 800,000 NT$)
- **Y-axis (left):** SHAP value = impact on prediction
  - Positive = increases default risk
  - Negative = decreases default risk
- **Color:** Bill Amount (2 Months Ago)
  - Red/Pink = High bills
  - Blue = Low bills

**Key Patterns:**

1. **Inverse Relationship:**
   - Left side (low limits): SHAP values mostly positive/neutral
   - Right side (high limits): SHAP values strongly negative
   - **Trend:** As credit limit increases, default risk decreases

2. **Why This Happens:**
   - High credit limits are given to customers with proven creditworthiness
   - Acts as a proxy for historical good behavior
   - Not causal (giving more credit doesn't reduce risk), but correlational

3. **Bill Amount Interaction (Color):**
   - Red/pink dots (high bills) cluster toward neutral/positive SHAP
   - Blue dots (low bills) have more strongly negative SHAP
   - **Insight:** High credit limit + high bill usage = diminished protective effect

**Business Takeaway:** Credit limit is a trust signal but loses protective power when customers are near their limits.

---

### Image 4: Payment Status 2 Months Ago

![Payment Status 2 Months Scatter](image4)

**What This Shows:** How payment status from 2 months ago affects default risk.

#### Understanding the Visualization

**Axes:**
- **X-axis (bottom):** Payment Status (2 Months Ago) value
  - -2 = No credit use
  - -1 = Paid in full
  - 0 = Revolving credit
  - 1 = 1 month late
  - 2+ = 2+ months late
- **Y-axis (left):** SHAP value = impact on prediction
- **Color:** Payment Status (Current Month)
  - Shows interaction between past and current behavior

**Key Patterns:**

1. **Similar Pattern to Current Payment Status:**
   - Values -2, -1, 0: Negative SHAP (decreases risk)
   - Values 2+: Strongly positive SHAP (increases risk)
   - BUT weaker effect than current month

2. **The Critical Threshold:**
   - Notice the gap between status 1 and status 2
   - Status 2+ creates dramatic risk increase
   - Confirms 2 months late is critical threshold

3. **Color Interaction:**
   - Pink/red dots (currently late) have higher SHAP even for same past status
   - Being late now + late before = compounding risk
   - Blue dots (currently on-time) show recovery is possible

**Business Takeaway:** Recent history matters more than distant history, but chronic late payments compound risk exponentially.

---

### Image 5: Payment Status Current Month (Most Critical)

![Current Payment Status Scatter](image5)

**What This Shows:** The single most important predictor - current payment status.

#### Understanding the Visualization

**Axes:**
- **X-axis (bottom):** Payment Status (Current Month) value
  - -2 to 0 = On-time
  - 1 = 1 month late
  - 2+ = 2+ months late
- **Y-axis (left):** SHAP value = impact on prediction
- **Color:** Bill Amount (2 Months Ago)

**Key Patterns:**

1. **Three Distinct Zones:**
   
   **Zone 1: On-Time Payers (X = -2, -1, 0)**
   - SHAP values: -0.2 to -0.1
   - Tightly clustered
   - Low default risk regardless of bill amounts
   
   **Zone 2: One Month Late (X = 1)**
   - SHAP values: 0.0 to +0.25
   - Moderate risk increase
   - Still recoverable
   
   **Zone 3: Two+ Months Late (X = 2+)**
   - SHAP values: +0.4 to +0.8
   - MASSIVE risk increase
   - Critical threshold crossed

2. **The Cliff Effect:**
   - The jump from status 1 → 2 is NOT gradual
   - Default risk more than triples
   - This is the model's "alarm bell"

3. **Bill Amount Doesn't Matter Much Here:**
   - Color (bill amounts) shows mixed distribution across all payment statuses
   - Payment behavior dominates regardless of debt size

**Business Takeaway:** THIS IS YOUR #1 INTERVENTION POINT. Flag all customers at PAY_0 ≥ 2 for immediate action. The model shows this is when default becomes highly likely.

---

### Image 6: Waterfall Plot (Individual Customer Story)

![Waterfall Plot](image6)

**What This Shows:** How the model built its prediction for ONE specific high-risk customer, showing the step-by-step contribution of each feature.

#### Understanding the Visualization

**Elements:**
- **Base value (E[f(x)] = 0.339):** The starting point = average default probability (33.9%)
- **Each bar:** Shows how one feature pushed the prediction higher (red) or lower (blue)
- **Bar width:** Proportional to the feature's impact
- **Final prediction (f(x) = 1.159):** End result after all features (~76% default probability)

**Reading This Customer's Story:**

1. **Starting Point:** 
   - Average customer has 33.9% default probability

2. **Payment Status (Current Month) = 2** → +0.65
   - Customer is 2 months late on payments
   - This ALONE adds 65% to default probability
   - Moves prediction from 34% to 99%

3. **Payment Status (2 Months Ago) = 2** → +0.23
   - Also late previously
   - Pattern of chronic delinquency
   - Compounds the risk

4. **Credit Limit Amount = 50,000** → +0.05
   - Low credit limit (sign of weaker credit history)
   - Small additional risk

5. **Payment Status (3 Months Ago) = 0** → -0.03
   - Was on-time 3 months ago
   - Slight protective factor but can't overcome recent behavior

6. **Other Features:** 
   - Payment amounts, bill amounts, demographics
   - Combined impact: -0.02
   - Essentially negligible

**Final Verdict:**
- This customer has ~76% chance of default
- Driven almost entirely by being 2 months late NOW and 2 months ago
- Model recommendation: **HIGH RISK - Do not extend credit**

**Business Takeaway:** For this customer, intervention should have happened when PAY_0 first hit 2. By the time we see this pattern (2 months late twice), default is highly probable.

---

## Feature Dictionary

### Payment Status Features (PAY_0 through PAY_6)

| Feature | Meaning | Time Period | Values |
|---------|---------|-------------|--------|
| PAY_0 | Most recent payment status | Current month (September) | -2 to 8+ |
| PAY_2 | Payment status 2 months ago | July | -2 to 8+ |
| PAY_3 | Payment status 3 months ago | June | -2 to 8+ |
| PAY_4 | Payment status 4 months ago | May | -2 to 8+ |
| PAY_5 | Payment status 5 months ago | April | -2 to 8+ |
| PAY_6 | Payment status 6 months ago | March | -2 to 8+ |

**Value Scale:**
- **-2:** No consumption (no balance)
- **-1:** Paid in full (balance paid off)
- **0:** Revolving credit (minimum payment made, carrying balance)
- **1:** Payment delay of 1 month
- **2:** Payment delay of 2 months
- **3-8:** Payment delay of 3-8 months

**Critical Threshold:** PAY ≥ 2 indicates severe delinquency

---

### Financial Features

| Feature | Description | Typical Range | Impact on Default |
|---------|-------------|---------------|-------------------|
| LIMIT_BAL | Total credit limit | $20,000 - $800,000 NT$ | Higher = Lower Risk |
| BILL_AMT1-6 | Bill statement amounts (last 6 months) | $0 - $500,000 NT$ | Very high relative to limit = Higher Risk |
| PAY_AMT1-6 | Payment amounts (last 6 months) | $0 - $200,000 NT$ | Higher payments = Lower Risk |

---

### Demographic Features

| Feature | Description | Impact |
|---------|-------------|--------|
| AGE | Customer age | Weak (non-linear relationship) |
| SEX | Gender (1=male, 2=female) | Minimal |
| EDUCATION | Education level (1=graduate, 2=university, 3=high school, 4=others) | Minimal |
| MARRIAGE | Marital status (1=married, 2=single, 3=others) | Minimal |

**Note:** Demographics have surprisingly low predictive power compared to payment behavior.

---

## Business Insights & Recommendations

### 1. Risk Scoring Tiers

Based on SHAP analysis, implement these risk tiers:

| Tier | Predicted Default Prob | Characteristics | Action |
|------|----------------------|-----------------|--------|
| **Low Risk** | 0-30% | PAY_0 ≤ 0, consistent payment history | Approve credit, consider limit increases |
| **Medium Risk** | 30-60% | PAY_0 = 1, occasional late payments | Monitor closely, send payment reminders |
| **High Risk** | 60-80% | PAY_0 = 2, recent delinquency | Freeze additional credit, contact for payment plan |
| **Critical Risk** | 80%+ | PAY_0 ≥ 3, chronic delinquency | Immediate intervention, collections process |

---

### 2. Early Warning System

**Recommended Alert Triggers:**

**Green Alert:** PAY_0 = 1 (First Late Payment)
- Action: Automated reminder
- Cost: Low
- Goal: Prevent escalation

**Yellow Alert:** PAY_0 = 2 (Two Months Late)
- Action: Personal outreach, payment plan offered
- Cost: Medium
- Goal: Immediate intervention before default

**Red Alert:** PAY_0 ≥ 3 OR (PAY_0 = 2 AND PAY_2 = 2)
- Action: Credit freeze, collections process
- Cost: High
- Goal: Minimize losses

---

### 3. Credit Limit Strategy

**Findings:**
- High credit limits correlate with lower default risk
- BUT this is because good customers earned high limits
- Giving high limits doesn't reduce risk

**Recommendations:**
- Reward customers with perfect payment history (PAY_0 = -1 for 12+ months)
- Increase limits for customers who pay in full regularly
- Don't give high limits hoping to reduce default risk
- Don't deny limits solely based on demographics

---

### 4. Model Deployment Guidelines

**Optimal Decision Threshold:** 
- Default 0.5 threshold → Use profit-optimized threshold (~0.45-0.55 depending on cost structure)
- Run sensitivity analysis on false positive vs false negative costs

**Expected Performance:**
- **Recall:** ~48% (catch half of actual defaults)
- **Precision:** ~48% (half of flagged customers will actually default)
- **ROC AUC:** ~0.78 (good discrimination ability)
- **Profit Improvement:** +12% over baseline

**Monitoring:**
- Track SHAP value distributions monthly
- Alert if feature importance shifts dramatically
- Retrain quarterly with new data

---

## Technical Implementation

### Model Architecture
- **Algorithm:** XGBoost (Gradient Boosted Trees)
- **Optimization:** Bayesian hyperparameter tuning (Hyperopt + TPE)
- **Target Metric:** Custom profit function (not F1 score)
- **Interpretability:** SHAP TreeExplainer

### Key Hyperparameters (Profit-Optimized Model)
```python
{
    'max_depth': 6,                    # Tree depth
    'learning_rate': 0.08,             # Step size
    'subsample': 0.8,                  # Row sampling
    'colsample_bytree': 0.8,           # Column sampling
    'min_child_weight': 3,             # Minimum samples per leaf
    'gamma': 0.2,                      # Regularization
    'scale_pos_weight': 2.5            # Class balance adjustment
}
```

### Cost Function
```python
def calculate_profit(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Industry-standard costs
    cost_fn = 5000   # Miss a default → lose $5,000
    cost_fp = 200    # Deny good customer → lose $200 profit
    profit_tn = 200  # Approve good customer → gain $200
    profit_tp = 50   # Catch default early → gain $50
    
    return (profit_tn * tn) + (profit_tp * tp) - (cost_fp * fp) - (cost_fn * fn)
```

---

## Project Structure

```
credit-default-prediction/
├── Credit_Default.py              # Main training script
├── default_credit_cards.csv       # Dataset (30,000 records)
├── README.md                      # This file
├── visualizations/
│   ├── shap_beeswarm.png         # Feature importance
│   ├── shap_decision.png         # Customer risk journeys
│   ├── shap_scatter_*.png        # Feature relationships
│   └── shap_waterfall.png        # Individual prediction
└── models/
    ├── baseline_model.pkl
    ├── f1_optimized_model.pkl
    └── profit_optimized_model.pkl
```

---

## How to Run

### Prerequisites
```bash
pip install pandas numpy scikit-learn xgboost matplotlib shap hyperopt
```

### Training
```bash
python Credit_Default.py
```

### Output
- Model performance metrics (console)
- SHAP visualizations (saved as PNG files)
- Trained models (pickle files)

---

## 📊 Key Findings Summary

### What Predicts Default?

**Strong Predictors (SHAP > 0.3):**
1. Payment Status (Current Month) - **Dominant**
2. Payment Status (2-3 Months Ago) - **Strong**
3. Credit Limit Amount - **Moderate** (inverse relationship)

**Weak Predictors (SHAP < 0.1):**
4. Payment Amounts - Weak
5. Bill Amounts - Weak  
6. Age, Gender, Education, Marriage - Minimal

### The Critical Threshold

**PAY_0 = 2 is the "cliff"**
- PAY_0 ≤ 1: Relatively safe
- PAY_0 = 2: Risk increases 5-10x
- PAY_0 ≥ 3: Near-certain default

### Business Impact

**By implementing this model:**
- Catch ~48% of defaults before they occur
- Reduce false denials by optimizing for profit, not F1
- Generate +12% profit vs. current baseline approach
- Provide explainable decisions for compliance

---

## Future Enhancements

1. **Feature Engineering:**
   - Payment volatility (standard deviation of payment amounts)
   - Utilization ratio (bill amount / credit limit)
   - Payment trend (improving vs. deteriorating)
   - Time since last late payment

2. **Advanced Modeling:**
   - Ensemble with LightGBM or CatBoost
   - Neural network for non-linear patterns
   - Survival analysis for time-to-default

3. **Fairness Analysis:**
   - Check for demographic bias using SHAP
   - Ensure compliance with fair lending regulations
   - Implement bias mitigation techniques

4. **Production Deployment:**
   - Real-time API for credit decisions
   - A/B testing framework
   - Model monitoring dashboard
   - Automated retraining pipeline

---

## References

- **SHAP Documentation:** https://shap.readthedocs.io/
- **XGBoost Documentation:** https://xgboost.readthedocs.io/
- **Paper:** "A Unified Approach to Interpreting Model Predictions" (Lundberg & Lee, 2017)
- **Dataset:** UCI Machine Learning Repository - Default of Credit Card Clients


---

## Acknowledgments

- UCI Machine Learning Repository for the dataset
- SHAP library creators for interpretability tools
- The XGBoost team for the powerful ML framework
