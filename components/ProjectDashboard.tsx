import { useEffect, useMemo, useState } from "react";

type DashboardStats = {
  completedToday: number;
  updatedToday: number;
  createdToday: number;
  overdue: number;
};

type StatsCardProps = {
  title: string;
  value: number | string;
  description?: string;
};

const DASHBOARD_PROMPT = `
Return only a JSON object with numeric values for the keys
completedToday, updatedToday, createdToday, and overdue.
The values must describe the latest project task metrics and be
integers. Example: {"completedToday": 3, "updatedToday": 5, "createdToday": 2, "overdue": 1}.
`;

const resolveApiServer = (): string | undefined => {
  if (typeof window !== "undefined") {
    const fromWindow = (window as unknown as { API_SERVER?: string }).API_SERVER;
    if (fromWindow) {
      return fromWindow;
    }
  }

  const env =
    (typeof process !== "undefined" && process.env) ||
    (import.meta as unknown as { env?: Record<string, string | undefined> }).env;

  if (env) {
    return (
      env.NEXT_PUBLIC_API_SERVER ||
      env.VITE_API_SERVER ||
      env.API_SERVER ||
      env.REACT_APP_API_SERVER
    );
  }

  return undefined;
};

const apiServer = resolveApiServer();

const StatsCard = ({ title, value, description }: StatsCardProps) => (
  <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
    <div className="text-sm font-medium text-slate-500">{title}</div>
    <div className="mt-2 text-3xl font-semibold text-slate-900">{value}</div>
    {description ? (
      <div className="mt-1 text-xs text-slate-400">{description}</div>
    ) : null}
  </div>
);

const parseAnswer = (answer: unknown): DashboardStats | null => {
  if (!answer || typeof answer !== "string") {
    return null;
  }

  try {
    const parsed = JSON.parse(answer) as Partial<DashboardStats>;
    if (
      typeof parsed.completedToday === "number" &&
      typeof parsed.updatedToday === "number" &&
      typeof parsed.createdToday === "number" &&
      typeof parsed.overdue === "number"
    ) {
      return parsed as DashboardStats;
    }
  } catch (error) {
    // Fall through to regex parsing below.
  }

  const matches = answer.match(/\d+/g);
  if (matches && matches.length >= 4) {
    const [completedToday, updatedToday, createdToday, overdue] = matches
      .slice(0, 4)
      .map((value) => Number.parseInt(value, 10));

    if ([completedToday, updatedToday, createdToday, overdue].every(Number.isFinite)) {
      return { completedToday, updatedToday, createdToday, overdue };
    }
  }

  return null;
};

const ProjectDashboard = () => {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const endpoint = useMemo(() => {
    if (!apiServer) {
      return undefined;
    }

    return `${apiServer.replace(/\/$/, "")}/ask`;
  }, []);

  useEffect(() => {
    if (!endpoint) {
      setError("API server URL is not configured.");
      return;
    }

    const controller = new AbortController();
    let cancelled = false;

    const loadStats = async () => {
      setLoading(true);
      setError(null);

      try {
        const response = await fetch(endpoint, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            question: DASHBOARD_PROMPT.trim(),
            top_k: 3,
          }),
          signal: controller.signal,
        });

        if (!response.ok) {
          throw new Error(`Request failed with status ${response.status}`);
        }

        const payload: {
          ok?: boolean;
          answer?: unknown;
          error?: string;
        } = await response.json();

        if (payload.ok !== true) {
          throw new Error(payload.error || "Dashboard data request failed.");
        }

        const parsed = parseAnswer(payload.answer);
        if (!parsed) {
          throw new Error("Unable to parse dashboard statistics from the response.");
        }

        if (!cancelled) {
          setStats(parsed);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Unknown error");
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    };

    loadStats();

    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [endpoint]);

  return (
    <section className="space-y-6">
      <header className="space-y-1">
        <h1 className="text-2xl font-semibold text-slate-900">Project Dashboard</h1>
        <p className="text-sm text-slate-500">
          Overview of the most important project task metrics.
        </p>
      </header>

      {error ? (
        <div className="rounded-md border border-red-200 bg-red-50 p-4 text-sm text-red-700">
          {error}
        </div>
      ) : null}

      {loading ? (
        <div className="text-sm text-slate-500">Loading project statistics…</div>
      ) : null}

      {stats ? (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <StatsCard title="Completed Today" value={stats.completedToday} />
          <StatsCard title="Updated Today" value={stats.updatedToday} />
          <StatsCard title="Created Today" value={stats.createdToday} />
          <StatsCard title="Overdue" value={stats.overdue} />
        </div>
      ) : null}
    </section>
  );
};

export default ProjectDashboard;
