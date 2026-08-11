# My Thought Process: Churn Prediction FDE Assignment

I had three days, a dataset I had never seen before, and a stack I had only partially worked with. Here is how I actually approached it.

## Research First

I started by reading about churn prediction methods and comparing statistical approaches before writing a single line of code.

I used two AI models during research: Gemini 3.1 Pro and Claude Opus 4.8, to compare algorithm recommendations and build a learning plan. Then I used Claude Sonnet 4.6 with Claude Code to execute and review.

## The Algorithm Decision

I wanted to combine XGBoost with CoxPH. CoxPH is a survival analysis model that tells you not just whether someone will churn but when.

I thought that would be genuinely useful for a marketing platform because you could tier urgency in your campaigns. A customer likely to leave in 3 days gets a different offer than one likely to leave in 30.

But I was honest with myself. Three days was not enough time to implement both, test both, and explain both clearly. Even with AI assistance I want to understand what I build.

XGBoost with SHAP gives a clean explainable story. CoxPH is documented as a future enhancement with the reasoning intact.

## Things I Learned That Surprised Me

The 999.0 sentinel value. I initially thought using 0 for a customer with no prior session made sense. Then I realized 0 already means something: a session that lasted zero seconds, which can happen from system bugs or app crashes.

Using 0 would silently mix two completely different customer states. 999 is far enough from any real recency value that the model treats it as its own category. That distinction matters in production.

Temporal leakage. This one genuinely caught me off guard. I assumed you compute features and labels from the same data. Wrong.

If your label is "did they churn in the last 30 days" and your recency feature is also computed from that same 30-day window, the model learns from the future. You set a hard cutoff: features come from before that date, label comes from after. That boundary is the whole design.

## What I Added Beyond the Spec

The dashboard. I did not have to build a UI. But a number between 0 and 1 means nothing to most stakeholders. A red badge that says HIGH RISK with the top three reasons underneath means something. I built it because showing is always better than describing, and it saves a conversation in every demo.

CloudWatch observability. I am a believer that you should know about errors before your customers do. The alarm on 5 errors in 5 minutes is basic but it is there from day one. In a real deployment I would add latency percentiles, score distribution drift detection, and a weekly fairness metric report.

Then there is the real-time scoring idea I did not build but documented. Offline RFM stored in S3, synced to Redis. The mobile app queries the ML service during the billing or account screen. The model scores in real time, and if the score exceeds a threshold it triggers a coupon or discount workflow with a non-intrusive popup. Websockets would be cleaner but heavier. A Redis query is fast enough for this use case.

## How I Would Do This With a Team

I would not have built the synthetic data generator alone. I would have gone to the data scientist first and asked what the real behavioral differences look like between churned and retained customers. That conversation shapes everything.

In isolation I had to make assumptions and document them. With a team those assumptions become decisions made together.

If I had no team but needed domain expertise, I would build a specialized research agent, feed it literature on churn prediction in mobile marketing, and use it to pressure-test my feature engineering choices before writing code.

## On Using AI in This Project

I used Claude Code as a development accelerator, not a replacement for thinking. Every section was built prompt by prompt, reviewed before moving on.

Claude Code caught real errors I would have missed: the temporal leakage between CUTOFF and AS_OF, the F2 degeneracy when churn is the majority class, the Chouldechova impossibility result in the fairness analysis, and the threshold overshoot from tied tree scores.

In each case I had to understand the finding before I could decide what to do with it. That is the only way I know how to work with AI tools.

## What I Would Change With More Time

The front end. Every AI-generated UI looks the same. I would add number inputs alongside the sliders so users can type exact values. I would add a customer lookup by ID against a real database. I would make the SHAP visualization more narrative and less chart. I want stakeholders to read it, not interpret it.

Load testing and rate limiting under real traffic. I set 100 requests per minute per IP but I have not tested what the model actually does under sustained load. That is a gap I would close before any production deployment.

## Closing

This was three days of real engineering work and I am proud of what came out of it.
