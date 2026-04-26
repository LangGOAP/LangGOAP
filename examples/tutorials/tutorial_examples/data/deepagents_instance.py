"""Sample document for the DeepAgents integration tutorial.

A multi-paragraph text that exercises all four content-analysis actions
(topic extraction, sentiment analysis, summarization, recommendations).
"""

SAMPLE_DOCUMENT: str = """\
The adoption of cloud-native architectures has accelerated dramatically over \
the past two years.  Organizations that previously relied on monolithic \
deployments are migrating to microservices orchestrated by Kubernetes, \
benefiting from improved scalability and fault isolation.  However, the \
transition introduces operational complexity: observability stacks must cover \
distributed traces, service meshes add latency overhead, and configuration \
drift across hundreds of services becomes a real risk.

Security teams report mixed feelings.  Container image scanning and runtime \
policies have matured, yet supply-chain attacks targeting base images and \
third-party dependencies remain a top concern.  Zero-trust networking \
principles are gaining traction, but retrofitting them into brownfield \
environments is expensive and disruptive.

Developer experience is improving.  Platform engineering teams now offer \
self-service portals backed by internal developer platforms (IDPs) that \
abstract away infrastructure provisioning.  GitOps workflows—where the \
desired state of every environment is declared in version control—have \
reduced deployment failures and shortened mean-time-to-recovery.  \
Nonetheless, cognitive load on individual developers remains high, and \
many teams still struggle with debugging latency issues that span multiple \
services.

Looking ahead, industry analysts predict that AI-assisted operations \
(AIOps) will play a larger role in anomaly detection and automated \
remediation.  FinOps practices are also maturing, helping organizations \
correlate cloud spend with business value rather than treating \
infrastructure costs as an undifferentiated overhead.
"""
