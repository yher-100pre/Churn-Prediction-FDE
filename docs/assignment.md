**LOCALYTICS · FDE**

**Churn Prediction Service — Forward Deployment Engineering**

| Time window | 4 days from account invite (hard deadline) — hours worked are your call |
| :---- | :---- |
| **Format** | Python, Terraform, sample dataset provided |
| **Focus areas** | Feature engineering, production deployment, platform integration, operational readiness |

# **Scenario**

We're providing a sample of **raw events only** (sessions, purchases, pushes, etc.) in JSON. We are **not** providing a pre-built feature set.

Your first modeling step is feature engineering: transform raw events into **RFM (Recency, Frequency, Monetary)** features, and use that table as training data. You may add additional features if you justify them.

This mirrors the first prediction capability we want live in production: churn probability feeding campaign audience selection. Your job as a Forward Deployment Engineer is to design, build, and ship a production-ready path for that capability—including the raw→RFM pipeline—not a notebook experiment.

# **Sample dataset**

The sample dataset is [`data/events.json`](data/events.json) — **800 raw event rows** in JSON. Start with [`data/dataset_schema.json`](data/dataset_schema.json) for field definitions, event types, the as-of timestamp for windows, and the RFM expectation. See [`data/README.md`](data/README.md) for a short walkthrough.

This raw sample is intentionally small and meant to show you the event shapes, not to be your final training set. See [`data/README.md`](data/README.md) for guidance on scaling it up with synthetic data before you train or evaluate anything.

# **AWS interview account**

An AWS interview account will be created for you so you can complete this exercise. Access is limited to **4 days**, starting when the AWS account invite is sent.

This account is shared and subject to **strict budget limits**. Size your infrastructure conservatively (e.g. smallest viable EKS node group, avoid always-on endpoints if batch/serverless scoring would do) — resources that trip the budget limit may be suspended or reclaimed.

This account is meant to be used, not just referenced in a diagram. We expect your Terraform to be `apply`'d there and your service actually deployed and reachable — a plan-only submission is not sufficient. Since account access lapses after 4 days, include in your submission whatever evidence (example requests/responses, logs, screenshots) an interviewer would need to confirm it worked, in case access has expired by the time we review it.

Localytics is happy to schedule a short meeting to confirm your credentials are working and to provide additional context about the project if needed. Reach out if you want to set that up.

# **Objectives**

* Engineer features from raw events: build an RFM (Recency, Frequency, Monetary) training table (plus any justified extras), and keep that transform reproducible in your pipeline.

* Scale the sample into a training-sized dataset: write a reproducible script that generates additional synthetic customers/events preserving the real sample's statistical structure (event mix, timing, amounts), then justify the size and assumptions you chose. Don't train or evaluate directly on the 80-customer raw sample alone.

* Design and implement a churn prediction service that can run in a production-like environment on the provided AWS account.

* Establish a clear baseline for prediction quality (a simple heuristic or rule is fine) and show your approach improves on it with metrics that fit a churn / campaign-selection use case (not just accuracy).

* Make prediction outcomes understandable to product stakeholders: which signals drive a high churn score, and why those drivers are credible.

* Check for uneven performance across subgroups implied by profile or engineered features. If gaps exist, say what you'd do about them before shipping.

* Show how the service integrates with the platform (ingress/gateway, auth, observability) and how you would operate it after deploy.

# **How to submit**

**Fork this repository** and put your complete solution in the fork. Your fork should include all of the code, infrastructure, and diagrams for the exercise—do not submit materials only as separate attachments or links outside the repo.

# **What to submit**

1. **Feature Engineering & Reproducible Pipeline**: Code that transforms raw events → RFM (and any additional features), then trains or refreshes scoring and reproduces results.  
2. **Service Code**: Code to build and run the churn prediction service in a production-like setup.  
3. **Evaluation Metrics**: Show your metrics compared to your baseline (a simple heuristic). Include reasoning for why your chosen metrics are appropriate for a churn problem (considering class imbalance and the business cost of false negatives vs. false positives).  
4. **Explainability Output:** Show which signals drive predictions (plots or equivalent) and include a concise interpretation for a non-technical stakeholder.  
5. **Bias/Fairness Note:** Detail what you checked, what you discovered, and what recommendations you have if any performance gaps exist.  
6. **Architecture Diagram:** Illustrate your proposed production solution, including where feature engineering runs.  
7. **Infrastructure:** Include Terraform (or equivalent) for the infrastructure your architecture needs (e.g. EKS ingress/route configuration).

# **What we're evaluating**

* Forward Deployment Engineering judgment: can you take an ambiguous customer problem and ship a credible production path?

* Quality of feature engineering from raw events (RFM definitions, leakage awareness, reproducibility)—not reliance on a handed feature matrix.

* Whether your evaluation approach fits a churn / campaign-selection problem (class imbalance, cost of false negatives vs. false positives).

* Whether explainability, fairness checks, and operational concerns are built into the design—not bolted on at the end.

* How clearly you communicate system behavior and trade-offs to product and platform stakeholders.

* Quality of architecture, infrastructure-as-code, and production readiness (auth, rate limiting, observability, failure modes).

# **Architecture Requirements**

Provide a detailed architecture diagram of the proposed solution. This must include how the service fits into the platform’s existing ingress/gateway stack and authentication model.

# **Technology Stack**

The following technologies are allowed and preferred:

* Well-known programming languages (Python and Terraform encouraged)  
* Spark/EMR/EKS  
* SageMaker  
* S3 (data lake for raw events and engineered features)  
* IAM  
* Athena  
* Bedrock  
* Airflow  
* Parquet  
* Provide Terraform for the infrastructure (EKS ingress/route configuration).

While these are preferred, alternatives can be used if a valid reason is provided. Your solution does not require you to use all of these, just the ones your architecture needs.

# **Production Readiness**

Explain how your service handles service-to-service authentication, how you would implement rate limiting, and how you would expose observability metrics (e.g., latency, error rate) back to the centralized platform dashboard.

# **Notes & FDE Mindset**

* **Questions are Encouraged:** We prefer clarity over guesswork. If you hit a wall, need clarification on constraints, or want to discuss a strategic trade-off, reach out—there is no penalty for asking.  
* **Think Production-First:** Do not stop at a working scorer; design a system that includes the raw→RFM path. We evaluate your solution on how it integrates into a live production environment. Consider failure modes, latency, cold-start users with sparse events, and how you would detect data quality or score drift once deployed.  
* **The "So What?" Factor:** As an FDE, you are the bridge between technical capability and customer outcomes. For every technical decision (RFM definitions, scoring approach, infrastructure setup), be prepared to answer: *How does this specifically improve the outcome for the end user?*  
* **Explainable Trade-offs:** You will often need to balance sophistication against reliability and transparency. Explicitly state your trade-offs; a theoretically "perfect" scorer is often inferior to a reliable, explainable service the business can trust and act on.  
* **Operational Rigor:** Prioritize maintainable, modular code and infrastructure over "quick-and-dirty" scripts. As you architect your solution, think about how an operator would support this system at 3:00 AM.
* **Looking Ahead:** This exercise is deliberately scoped to the pipeline and service, not agents. In the follow-up conversation, come ready to walk through your solution and discuss where agentic components could extend it — no need to build or design that now.

*Questions during the exercise? Reach out any time; we'd rather you ask than guess. There is no penalty for asking clarifying questions.*
