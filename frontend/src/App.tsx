/**
 * Multi-Agent ML Pipeline Frontend
 * Streamlit-classic and Modern AI Studio themes, wired to the real FastAPI backend.
 */

import React, { useCallback, useEffect, useRef, useState } from 'react';
import { DatasetSummary, PipelineConstraints, PipelineRun, ProblemSpec } from './types';
import { StreamlitSidebar } from './components/StreamlitSidebar';
import { Clarification, NewRunView } from './components/NewRunView';
import { LiveRunView } from './components/LiveRunView';
import { RunHistoryView } from './components/RunHistoryView';
import { ArchitectureView } from './components/ArchitectureView';
import { api, errorMessage } from './services/api';
import { isTerminal, toDatasetSummary, toPipelineRun } from './services/adapter';
import type { WireCreateRunResponse, WireProblemSpec } from './services/apiTypes';

type View = 'new_run' | 'live_run' | 'history' | 'architecture';
type Theme = 'streamlit' | 'modern';

const POLL_MS = 1500;
const HISTORY_LIMIT = 30;

const FALLBACK_DATASET: DatasetSummary = toDatasetSummary(
  {
    source: 'builtin:breast_cancer',
    label: 'Breast Cancer Diagnostic (Wisconsin)',
    description:
      'Real-world clinical benchmark with 30 continuous numeric features computed from digitized FNA images of breast masses.',
    task_type: 'classification'
  },
  null
);

const VIEWS: View[] = ['new_run', 'live_run', 'history', 'architecture'];

/** Deep link / reload support: #history, #live_run, ... */
function viewFromHash(): View {
  const h = window.location.hash.replace('#', '');
  return (VIEWS as string[]).includes(h) ? (h as View) : 'new_run';
}

function loadTheme(): Theme {
  try {
    return localStorage.getItem('ui-theme') === 'streamlit' ? 'streamlit' : 'modern';
  } catch {
    return 'modern';
  }
}

export default function App() {
  const [uiTheme, setUiTheme] = useState<Theme>(loadTheme);
  const [currentView, setCurrentView] = useState<View>(viewFromHash);
  const [apiConnected, setApiConnected] = useState(true);

  // Datasets: the catalog from GET /datasets, plus any uploads; schema fills in lazily via peek.
  const [datasets, setDatasets] = useState<DatasetSummary[]>([FALLBACK_DATASET]);
  const [selectedId, setSelectedId] = useState<string>(FALLBACK_DATASET.id);
  const [peeked, setPeeked] = useState<Record<string, DatasetSummary>>({});
  const [peekLoading, setPeekLoading] = useState(true);
  const [peekError, setPeekError] = useState<string | null>(null);

  const [constraints, setConstraints] = useState<PipelineConstraints>({
    max_loop_backs: 3,
    max_tuning_trials: 15,
    cv_folds: 3
  });

  const [currentRun, setCurrentRun] = useState<PipelineRun | null>(null);
  const [runsHistory, setRunsHistory] = useState<PipelineRun[]>([]);
  const pollToken = useRef(0);

  const selectedDataset =
    peeked[selectedId] ?? datasets.find((d) => d.id === selectedId) ?? datasets[0];

  // ----- theme: class-based dark mode so the toggle (not the OS) drives `dark:` styles -----
  useEffect(() => {
    document.documentElement.classList.toggle('dark', uiTheme === 'modern');
    try {
      localStorage.setItem('ui-theme', uiTheme);
    } catch {
      /* storage unavailable: theme just won't persist */
    }
  }, [uiTheme]);

  // ----- API health -----
  useEffect(() => {
    let cancelled = false;
    const check = () =>
      api
        .health()
        .then(() => !cancelled && setApiConnected(true))
        .catch(() => !cancelled && setApiConnected(false));
    check();
    const id = setInterval(check, 10_000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  // ----- dataset catalog -----
  useEffect(() => {
    api
      .listDatasets()
      .then((items) => {
        if (items.length === 0) return;
        const list = items.map((i) => toDatasetSummary(i, null));
        setDatasets((prev) => [...list, ...prev.filter((p) => !list.some((l) => l.id === p.id) && p.id.startsWith('uploads'))]);
        setSelectedId(list[0].id);
      })
      .catch(() => {
        /* keep the fallback; the peek request below surfaces the real error */
      });
  }, []);

  // ----- peek the selected dataset (schema only, never rows) -----
  useEffect(() => {
    if (peeked[selectedId]) {
      setPeekLoading(false);
      setPeekError(null);
      return;
    }
    const base = datasets.find((d) => d.id === selectedId);
    if (!base) return;
    let cancelled = false;
    setPeekLoading(true);
    setPeekError(null);
    api
      .peekDataset(selectedId)
      .then((p) => {
        if (cancelled) return;
        const full = toDatasetSummary(
          { source: base.id, label: base.name, description: base.description, task_type: base.suggested_task },
          p
        );
        setPeeked((prev) => ({ ...prev, [selectedId]: full }));
      })
      .catch((e) => !cancelled && setPeekError(errorMessage(e)))
      .finally(() => !cancelled && setPeekLoading(false));
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedId, datasets]);

  // ----- run tracking: poll GET /runs/{id} until it reaches a terminal status -----
  const trackRun = useCallback(async (runId: string) => {
    const token = ++pollToken.current;
    while (pollToken.current === token) {
      try {
        const run = toPipelineRun(await api.getRun(runId));
        if (pollToken.current !== token) return;
        setCurrentRun(run);
        setRunsHistory((prev) => [run, ...prev.filter((r) => r.run_id !== run.run_id)]);
        if (isTerminal(run)) return;
      } catch {
        /* transient failure (backend restarting, etc.): keep polling */
      }
      await new Promise((r) => setTimeout(r, POLL_MS));
    }
  }, []);

  useEffect(() => () => void (pollToken.current += 1), []);

  const startTracking = (runId: string) => {
    setCurrentView('live_run');
    void trackRun(runId);
  };

  const loadHistory = useCallback(async () => {
    try {
      const summaries = await api.listRuns();
      const newest = [...summaries]
        .sort((a, b) => b.updated_at.localeCompare(a.updated_at))
        .slice(0, HISTORY_LIMIT);
      const details = await Promise.all(newest.map((s) => api.getRun(s.run_id).catch(() => null)));
      setRunsHistory(details.filter((d) => d !== null).map((d) => toPipelineRun(d)));
    } catch {
      /* leave whatever we already have */
    }
  }, []);

  useEffect(() => {
    void loadHistory();
  }, [loadHistory]);

  useEffect(() => {
    if (window.location.hash !== `#${currentView}`) window.location.hash = currentView;
  }, [currentView]);

  useEffect(() => {
    if (currentView === 'history') void loadHistory();
  }, [currentView, loadHistory]);

  // Opening Live Run with nothing selected shows the most recent run.
  useEffect(() => {
    if (currentView === 'live_run' && !currentRun && runsHistory.length > 0) {
      setCurrentRun(runsHistory[0]);
    }
  }, [currentView, currentRun, runsHistory]);

  // ----- launching -----
  const toWireSpec = (spec: ProblemSpec): WireProblemSpec => ({
    task_type: spec.task_type,
    target_column: spec.target_column,
    success_metric: spec.success_metric,
    metric_threshold: spec.metric_threshold,
    data_source: spec.data_source,
    constraints: {
      max_training_seconds: null,
      max_tuning_trials: spec.constraints.max_tuning_trials,
      interpretability_required: false,
      max_loop_backs: spec.constraints.max_loop_backs
    },
    notes: null
  });

  const handleLaunchStructured = async (spec: ProblemSpec) => {
    const res: WireCreateRunResponse = await api.createRun({ problem_spec: toWireSpec(spec) });
    if (res.status === 'needs_clarification') throw new Error(res.question_for_user);
    startTracking(res.run_id);
  };

  const handleLaunchContext = async (context: string): Promise<Clarification | null> => {
    const res = await api.createRun({
      context,
      file_path: selectedDataset.id,
      constraints: {
        max_tuning_trials: constraints.max_tuning_trials,
        max_loop_backs: constraints.max_loop_backs
      }
    });
    if (res.status === 'needs_clarification') {
      return { question_for_user: res.question_for_user, missing_fields: res.missing_fields };
    }
    startTracking(res.run_id);
    return null;
  };

  const handleUploadFile = async (file: File) => {
    const up = await api.uploadDataset(file);
    const summary = toDatasetSummary(
      {
        source: up.file_path,
        label: up.filename,
        description: `Uploaded file (${(up.size_bytes / 1024).toFixed(1)} KB). Only this schema summary is shown to the agents, never raw rows.`,
        task_type: null
      },
      null
    );
    setDatasets((prev) => [summary, ...prev]);
    setSelectedId(summary.id);
  };

  const handleSelectRun = (run: PipelineRun) => {
    setCurrentRun(run);
    setCurrentView('live_run');
    if (!isTerminal(run)) void trackRun(run.run_id);
  };

  const handleReRun = () => {
    if (!currentRun) return;
    void handleLaunchStructured(currentRun.problem_spec).catch((e) => console.error(errorMessage(e)));
  };

  const isStreamlit = uiTheme === 'streamlit';

  return (
    <div className={`min-h-screen md:h-screen md:overflow-hidden flex flex-col md:flex-row transition-colors duration-200 ${
      isStreamlit
        ? 'bg-[#ffffff] text-[#31333f] font-sans'
        : 'bg-slate-950 text-slate-100 font-sans'
    }`}>
      <StreamlitSidebar
        currentView={currentView}
        onSelectView={setCurrentView}
        uiTheme={uiTheme}
        onToggleTheme={setUiTheme}
        datasets={datasets.map((d) => peeked[d.id] ?? d)}
        selectedDataset={selectedDataset}
        onSelectDataset={(ds) => setSelectedId(ds.id)}
        constraints={constraints}
        onUpdateConstraints={setConstraints}
        apiConnected={apiConnected}
        hasActiveRun={currentRun?.status === 'running' || currentRun?.status === 'looping_back'}
      />

      <main className="flex-1 flex flex-col min-w-0 md:h-screen">
        <header className={`h-14 border-b flex items-center px-6 md:px-8 shrink-0 ${
          isStreamlit
            ? 'bg-[#ffffff] border-slate-200 text-slate-800'
            : 'bg-slate-900 border-slate-800 text-slate-200'
        }`}>
          <div className="mx-auto flex w-full max-w-6xl items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="text-xl">⚡</span>
            <div className="flex items-center gap-2">
              <span className="font-bold text-sm tracking-tight">
                Multi-Agent ML Pipeline
              </span>
              <span className="hidden sm:inline-block text-[11px] px-2 py-0.5 rounded-full bg-slate-100 dark:bg-slate-800 font-mono text-slate-600 dark:text-slate-400">
                API · {apiConnected ? 'Live' : 'Offline'}
              </span>
            </div>
          </div>

          <div className="flex items-center gap-3 text-xs">
            <div className="hidden sm:flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-slate-100 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 font-mono text-[11px]">
              <span className="text-slate-400">UI Mode:</span>
              <strong className={isStreamlit ? 'text-[#ff4b4b]' : 'text-indigo-400'}>
                {isStreamlit ? 'Streamlit Classical' : 'Modern ML Studio'}
              </strong>
            </div>

            <button
              onClick={() => setUiTheme(uiTheme === 'streamlit' ? 'modern' : 'streamlit')}
              className={`px-3 py-1 rounded-md font-semibold text-xs transition-all cursor-pointer ${
                isStreamlit
                  ? 'bg-slate-100 hover:bg-slate-200 text-slate-800 border border-slate-300'
                  : 'bg-indigo-600/80 hover:bg-indigo-600 text-white'
              }`}
            >
              Switch to {uiTheme === 'streamlit' ? 'AI Studio' : 'Streamlit'}
            </button>
          </div>
          </div>
        </header>

        <div className="flex-1 p-6 md:p-8 md:overflow-y-auto">
          {currentView === 'new_run' && (
            <NewRunView
              dataset={selectedDataset}
              constraints={constraints}
              uiTheme={uiTheme}
              peekLoading={peekLoading}
              peekError={peekError}
              onUploadFile={handleUploadFile}
              onLaunchStructured={handleLaunchStructured}
              onLaunchContext={handleLaunchContext}
            />
          )}

          {currentView === 'live_run' && (
            <LiveRunView run={currentRun} uiTheme={uiTheme} onReRun={handleReRun} />
          )}

          {currentView === 'history' && (
            <RunHistoryView runs={runsHistory} onSelectRun={handleSelectRun} uiTheme={uiTheme} />
          )}

          {currentView === 'architecture' && <ArchitectureView uiTheme={uiTheme} />}
        </div>
      </main>
    </div>
  );
}
