---
title: RootCite
emoji: 🛠️
colorFrom: yellow
colorTo: gray
sdk: docker
app_port: 8000
pinned: false
short_description: Evidence-cited root-cause analysis for production alerts
---

# RootCite live demo

Evidence-cited root-cause analysis for production alerts. Pick an incident, and RootCite
searches runbooks and postmortems, checks deploys, metrics, and logs through tool calls,
and returns a diagnosis that cites its evidence, with guardrails against prompt injection
and human approval for risky actions.

This Space is deployed automatically from the GitHub repository by GitHub Actions. It runs
on sample telemetry for five fictional services.
