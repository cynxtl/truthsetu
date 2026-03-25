import { useState, useEffect, useCallback } from "react";

const API = "http://127.0.0.1:8000/api/v1";

const VERDICT_CONFIG = {
  FALSE:       { color: "#ff6467", bg: "rgba(255,100,103,0.12)", icon: "✕", label: "FALSE" },
  CONFIRMED:   { color: "#a1a1a1", bg: "rgba(161,161,161,0.12)", icon: "✓", label: "CONFIRMED" },
  UNVERIFIED:  { color: "#737373", bg: "rgba(115,115,115,0.12)", icon: "?", label: "UNVERIFIED" },
  UNVERIFIABLE:{ color: "#737373", bg: "rgba(115,115,115,0.12)", icon: "?", label: "UNVERIFIED" },
  TRUE:        { color: "#a1a1a1", bg: "rgba(161,161,161,0.12)", icon: "✓", label: "CONFIRMED" },
  NO_INFO:     { color: "#525252", bg: "rgba(82,82,82,0.12)",   icon: "–", label: "NO INFO" },
  PENDING:     { color: "#525252", bg: "rgba(82,82,82,0.12)",   icon: "…", label: "PENDING" },
};

function useApi(endpoint, interval = 0) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  const fetch_ = useCallback(async () => {
    try {
      const r = await fetch(`${API}${endpoint}`);
      if (r.ok) setData(await r.json());
    } catch { /* backend not connected */ }
    finally { setLoading(false); }
  }, [endpoint]);

  useEffect(() => {
    fetch_();
    if (interval > 0) {
      const t = setInterval(fetch_, interval);
      return () => clearInterval(t);
    }
  }, [fetch_, interval]);

  return { data, loading, refetch: fetch_ };
}

function GlassCard({ children, className = "", style = {} }) {
  return (
    <div className={`glass-card ${className}`} style={style}>
      {children}
    </div>
  );
}

function StatBubble({ value, label, color = "#fff", sublabel }) {
  return (
    <div className="stat-bubble">
      <div className="stat-value" style={{ color }}>{value ?? "—"}</div>
      <div className="stat-label">{label}</div>
      {sublabel && <div className="stat-sublabel">{sublabel}</div>}
    </div>
  );
}

function VerdictBadge({ verdict }) {
  const cfg = VERDICT_CONFIG[verdict] || VERDICT_CONFIG.PENDING;
  return (
    <span className="verdict-badge" style={{
      color: cfg.color,
      background: cfg.bg,
      border: `1px solid ${cfg.color}33`,
    }}>
      <span className="verdict-icon">{cfg.icon}</span>
      {cfg.label}
    </span>
  );
}

function ClaimCard({ claim, index }) {
  const [expanded, setExpanded] = useState(false);
  const cfg = VERDICT_CONFIG[claim.verdict] || VERDICT_CONFIG.PENDING;

  const time = claim.verified_at
    ? new Date(claim.verified_at).toLocaleTimeString("en-IN", {
        hour: "2-digit", minute: "2-digit"
      })
    : "—";

  return (
    <div
      className="claim-card"
      style={{
        "--accent": cfg.color,
        animationDelay: `${index * 0.05}s`,
      }}
      onClick={() => setExpanded(!expanded)}
    >
      <div className="claim-card-header">
        <div className="claim-left">
          <div
            className="verdict-dot"
            style={{ background: cfg.color, boxShadow: `0 0 8px ${cfg.color}` }}
          />
          <div className="claim-text">{claim.claim || "—"}</div>
        </div>
        <div className="claim-right">
          <VerdictBadge verdict={claim.verdict} />
          <div className="claim-time">{time}</div>
          <div className="claim-expand">{expanded ? "↑" : "↓"}</div>
        </div>
      </div>

      {expanded && (
        <div className="claim-detail">
          {claim.reasoning && (
            <div className="detail-row">
              <span className="detail-label">Sources say</span>
              <span className="detail-value">{claim.reasoning}</span>
            </div>
          )}
          {claim.sources?.length > 0 && (
            <div className="detail-row">
              <span className="detail-label">Source</span>
              <span className="detail-value source-chip">
                {claim.sources[0]}
              </span>
            </div>
          )}
          {claim.platform && (
            <div className="detail-row">
              <span className="detail-label">Platform</span>
              <span className="detail-value">{claim.platform}</span>
            </div>
          )}
          {claim.from_cache && (
            <div className="detail-row">
              <span className="detail-label">Cache</span>
              <span className="detail-value" style={{ color: "#a1a1a1" }}>
                ⚡ Instant cache hit
              </span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function PulseRing() {
  return (
    <div className="pulse-ring-wrap">
      <div className="pulse-ring r1" />
      <div className="pulse-ring r2" />
      <div className="pulse-ring r3" />
      <div className="pulse-dot" />
    </div>
  );
}

function Header({ stats }) {
  return (
    <header className="header">
      <div className="header-brand">
        <div>
          <h1 className="brand-name">TruthSetu</h1>
          <p className="brand-tagline">Crisis Misinformation Intelligence</p>
        </div>
      </div>
      <div className="header-status">
        <PulseRing />
        <span className="status-text">Live</span>
      </div>
    </header>
  );
}

function StatsBar({ stats, learn }) {
  return (
    <GlassCard className="stats-bar">
      <StatBubble
        value={stats?.claims_today}
        label="Claims Today"
        color="#fafafa"
      />
      <div className="stat-divider" />
      <StatBubble
        value={stats?.false_claims_caught}
        label="False Claims"
        color="#ff6467"
      />
      <div className="stat-divider" />
      <StatBubble
        value={stats?.corrections_deployed}
        label="Corrections Sent"
        color="#a1a1a1"
      />
      <div className="stat-divider" />
      <StatBubble
        value={learn?.total_templates}
        label="Templates"
        color="#737373"
        sublabel={`${learn?.auto_learned || 0} auto-learned`}
      />
      <div className="stat-divider" />
      <StatBubble
        value={stats?.languages_active}
        label="Languages"
        color="#737373"
      />
      <div className="stat-divider" />
      <StatBubble
        value={stats?.avg_response_minutes
          ? `${stats.avg_response_minutes}s`
          : "—"}
        label="Avg Response"
        color="#737373"
      />
    </GlassCard>
  );
}

function AgentStatus() {
  const agents = [
    { name: "SCOUT",     icon: "◎", desc: "Claim extraction" },
    { name: "VERIFY",    icon: "◉", desc: "RAG + Web search" },
    { name: "TRANSLATE", icon: "◈", desc: "Hindi · Marathi" },
    { name: "DEPLOY",    icon: "◍", desc: "WhatsApp · Twilio" },
    { name: "LEARN",     icon: "◐", desc: "Template storage" },
  ];

  return (
    <GlassCard className="agent-panel">
      <div className="panel-title">Agent Pipeline</div>
      <div className="agent-list">
        {agents.map((a, i) => (
          <div key={a.name} className="agent-row" style={{ animationDelay: `${i * 0.1}s` }}>
            <div className="agent-icon" style={{ color: "#a1a1a1" }}>{a.icon}</div>
            <div className="agent-info">
              <div className="agent-name">{a.name}</div>
              <div className="agent-desc">{a.desc}</div>
            </div>
            <div className="agent-status-dot" />
          </div>
        ))}
      </div>
    </GlassCard>
  );
}

function SchedulerStatus() {
  const { data } = useApi("/scheduler/status", 60000);

  if (!data?.jobs) return null;

  return (
    <GlassCard className="scheduler-panel">
      <div className="panel-title">Scheduler</div>
      <div className="scheduler-list">
        {data.jobs.map((job) => {
          const next = new Date(job.next_run);
          const mins = Math.max(
            0,
            Math.round((next - Date.now()) / 60000)
          );
          return (
            <div key={job.id} className="scheduler-row">
              <div className="scheduler-name">{job.name}</div>
              <div className="scheduler-next">
                {mins === 0 ? "now" : `${mins}m`}
              </div>
            </div>
          );
        })}
      </div>
    </GlassCard>
  );
}

function ClaimsFeed({ claims, loading, total, onRefresh }) {
  return (
    <GlassCard className="claims-panel">
      <div className="panel-header">
        <div className="panel-title">
          Verified Claims
          {total > 0 && <span className="total-badge">{total}</span>}
        </div>
        <button className="refresh-btn" onClick={onRefresh}>↻</button>
      </div>

      {loading && (
        <div className="loading-state">
          <div className="loading-spinner" />
          <span>Loading claims...</span>
        </div>
      )}

      {!loading && (!claims || claims.length === 0) && (
        <div className="empty-state">
          <div className="empty-icon">◎</div>
          <div>No claims verified yet</div>
          <div className="empty-sub">
            Send a WhatsApp message to the Twilio sandbox to get started
          </div>
        </div>
      )}

      {!loading && claims?.length > 0 && (
        <div className="claims-list">
          {claims.map((claim, i) => (
            <ClaimCard
              key={claim._id || i}
              claim={claim}
              index={i}
            />
          ))}
        </div>
      )}
    </GlassCard>
  );
}

function VerdictDistribution({ claims }) {
  if (!claims?.length) return null;

  const counts = claims.reduce((acc, c) => {
    const v = c.verdict || "PENDING";
    acc[v] = (acc[v] || 0) + 1;
    return acc;
  }, {});

  const total = claims.length;
  const items = Object.entries(counts)
    .sort((a, b) => b[1] - a[1]);

  return (
    <GlassCard className="dist-panel">
      <div className="panel-title">Verdict Distribution</div>
      <div className="dist-list">
        {items.map(([verdict, count]) => {
          const cfg = VERDICT_CONFIG[verdict] || VERDICT_CONFIG.PENDING;
          const pct = Math.round((count / total) * 100);
          return (
            <div key={verdict} className="dist-row">
              <div className="dist-label">
                <span style={{ color: cfg.color }}>{cfg.label}</span>
                <span className="dist-count">{count}</span>
              </div>
              <div className="dist-bar-wrap">
                <div
                  className="dist-bar"
                  style={{
                    width: `${pct}%`,
                    background: cfg.color,
                    boxShadow: `0 0 8px ${cfg.color}66`,
                  }}
                />
              </div>
              <div className="dist-pct">{pct}%</div>
            </div>
          );
        })}
      </div>
    </GlassCard>
  );
}

export default function App() {
  const { data: stats, refetch: refetchStats } = useApi("/stats", 15000);
  const { data: claimsData, loading: claimsLoading, refetch: refetchClaims } =
    useApi("/claims?limit=20", 15000);
  const { data: learn } = useApi("/learn/stats", 30000);

  const claims = claimsData?.claims || [];
  const total  = claimsData?.total || 0;

  const handleRefresh = () => {
    refetchStats();
    refetchClaims();
  };

  return (
    <>
      <div className="bg-orbs">
        <div className="orb orb1" />
        <div className="orb orb2" />
        <div className="orb orb3" />
      </div>

      <div className="app">
        <Header stats={stats} />
        <StatsBar stats={stats} learn={learn} />

        <div className="main-grid">
          <div className="left-col">
            <AgentStatus />
            <SchedulerStatus />
            <VerdictDistribution claims={claims} />
          </div>
          <div className="right-col">
            <ClaimsFeed
              claims={claims}
              loading={claimsLoading}
              total={total}
              onRefresh={handleRefresh}
            />
          </div>
        </div>

        <footer className="footer">
          <span>TruthSetu · MIT-ADT University · Group TYAIA0310</span>
          <span className="footer-sep">·</span>
          <span>Sem VI · AI Multi-Agent Crisis Misinformation Detection</span>
        </footer>
      </div>
    </>
  );
}
