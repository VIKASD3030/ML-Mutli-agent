import React from 'react';
import {
  Sparkles,
  PlayCircle,
  Clock,
  BookOpen,
  Settings,
  Cpu,
  Database,
  Palette,
  CheckCircle,
  AlertCircle
} from 'lucide-react';
import { DatasetSummary, PipelineConstraints } from '../types';

interface StreamlitSidebarProps {
  currentView: 'new_run' | 'live_run' | 'history' | 'architecture';
  onSelectView: (view: 'new_run' | 'live_run' | 'history' | 'architecture') => void;
  uiTheme: 'streamlit' | 'modern';
  onToggleTheme: (theme: 'streamlit' | 'modern') => void;
  datasets: DatasetSummary[];
  selectedDataset: DatasetSummary;
  onSelectDataset: (ds: DatasetSummary) => void;
  constraints: PipelineConstraints;
  onUpdateConstraints: (c: PipelineConstraints) => void;
  apiConnected: boolean;
  hasActiveRun: boolean;
}

export const StreamlitSidebar: React.FC<StreamlitSidebarProps> = ({
  currentView,
  onSelectView,
  uiTheme,
  onToggleTheme,
  datasets,
  selectedDataset,
  onSelectDataset,
  constraints,
  onUpdateConstraints,
  apiConnected,
  hasActiveRun
}) => {
  const isStreamlitTheme = uiTheme === 'streamlit';

  return (
    <aside
      id="streamlit-sidebar"
      className={`w-full md:w-72 shrink-0 md:h-screen md:overflow-y-auto border-r flex flex-col p-4 select-none ${
        isStreamlitTheme
          ? 'bg-[#f0f2f6] border-slate-300 text-slate-800'
          : 'bg-slate-950 border-slate-800 text-slate-200'
      }`}
    >
      {/* Brand Header */}
      <div className="mb-6 pb-4 border-b border-slate-300/80 dark:border-slate-800">
        <div className="flex items-center gap-2">
          <span className="text-2xl">🎈</span>
          <div>
            <h1 className="text-sm font-bold tracking-tight">
              Multi-Agent ML
            </h1>
            <p className="text-[11px] opacity-70 font-mono">
              AutoML &amp; LLM Supervisors
            </p>
          </div>
        </div>

        {/* Theme Switcher Pill */}
        <div className="mt-3.5 p-1 rounded-lg bg-black/5 dark:bg-slate-900 border border-black/10 dark:border-slate-800 flex items-center justify-between text-xs">
          <span className="text-[11px] font-medium opacity-80 flex items-center gap-1.5 pl-1.5">
            <Palette className="w-3.5 h-3.5 text-[#ff4b4b]" />
            UI Theme
          </span>
          <div className="flex items-center gap-1">
            <button
              onClick={() => onToggleTheme('streamlit')}
              className={`px-2 py-0.5 rounded text-[11px] font-medium transition-all ${
                isStreamlitTheme
                  ? 'bg-[#ff4b4b] text-white shadow-xs'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              Streamlit
            </button>
            <button
              onClick={() => onToggleTheme('modern')}
              className={`px-2 py-0.5 rounded text-[11px] font-medium transition-all ${
                !isStreamlitTheme
                  ? 'bg-indigo-600 text-white shadow-xs'
                  : 'text-slate-600 hover:text-black'
              }`}
            >
              AI Studio
            </button>
          </div>
        </div>
      </div>

      {/* Navigation Radio (st.sidebar.radio) */}
      <div className="mb-6">
        <div className="text-[11px] font-bold uppercase tracking-wider opacity-60 mb-2 font-mono">
          Navigation
        </div>
        <div className="space-y-1">
          <button
            id="nav-new-run"
            onClick={() => onSelectView('new_run')}
            className={`w-full text-left px-3 py-2 rounded-lg text-xs font-medium flex items-center gap-2.5 transition-all cursor-pointer ${
              currentView === 'new_run'
                ? isStreamlitTheme
                  ? 'bg-white text-[#ff4b4b] font-bold shadow-xs border border-slate-200'
                  : 'bg-indigo-600 text-white font-bold'
                : isStreamlitTheme
                ? 'hover:bg-slate-200/70 text-slate-700'
                : 'hover:bg-slate-900 text-slate-400'
            }`}
          >
            <Sparkles className="w-4 h-4" />
            <span>➕ New Run</span>
          </button>

          <button
            id="nav-live-run"
            onClick={() => onSelectView('live_run')}
            className={`w-full text-left px-3 py-2 rounded-lg text-xs font-medium flex items-center justify-between transition-all cursor-pointer ${
              currentView === 'live_run'
                ? isStreamlitTheme
                  ? 'bg-white text-[#ff4b4b] font-bold shadow-xs border border-slate-200'
                  : 'bg-indigo-600 text-white font-bold'
                : isStreamlitTheme
                ? 'hover:bg-slate-200/70 text-slate-700'
                : 'hover:bg-slate-900 text-slate-400'
            }`}
          >
            <div className="flex items-center gap-2.5">
              <PlayCircle className="w-4 h-4" />
              <span>⚡ Live Run</span>
            </div>
            {hasActiveRun && (
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-ping" />
            )}
          </button>

          <button
            id="nav-history"
            onClick={() => onSelectView('history')}
            className={`w-full text-left px-3 py-2 rounded-lg text-xs font-medium flex items-center gap-2.5 transition-all cursor-pointer ${
              currentView === 'history'
                ? isStreamlitTheme
                  ? 'bg-white text-[#ff4b4b] font-bold shadow-xs border border-slate-200'
                  : 'bg-indigo-600 text-white font-bold'
                : isStreamlitTheme
                ? 'hover:bg-slate-200/70 text-slate-700'
                : 'hover:bg-slate-900 text-slate-400'
            }`}
          >
            <Clock className="w-4 h-4" />
            <span>📜 Run History</span>
          </button>

          <button
            id="nav-architecture"
            onClick={() => onSelectView('architecture')}
            className={`w-full text-left px-3 py-2 rounded-lg text-xs font-medium flex items-center gap-2.5 transition-all cursor-pointer ${
              currentView === 'architecture'
                ? isStreamlitTheme
                  ? 'bg-white text-[#ff4b4b] font-bold shadow-xs border border-slate-200'
                  : 'bg-indigo-600 text-white font-bold'
                : isStreamlitTheme
                ? 'hover:bg-slate-200/70 text-slate-700'
                : 'hover:bg-slate-900 text-slate-400'
            }`}
          >
            <BookOpen className="w-4 h-4" />
            <span>🔬 Architecture &amp; Spec</span>
          </button>
        </div>
      </div>

      {/* Dataset Selection (st.sidebar.selectbox) */}
      <div className="mb-6">
        <label className="text-[11px] font-bold uppercase tracking-wider opacity-60 mb-2 font-mono flex items-center gap-1">
          <Database className="w-3.5 h-3.5" />
          Active Benchmark
        </label>
        <select
          id="sidebar-dataset-select"
          value={selectedDataset.id}
          onChange={(e) => {
            const found = datasets.find(d => d.id === e.target.value);
            if (found) onSelectDataset(found);
          }}
          className={`w-full text-xs p-2 rounded-lg border outline-hidden transition-all ${
            isStreamlitTheme
              ? 'bg-white border-slate-300 text-slate-900 focus:border-[#ff4b4b]'
              : 'bg-slate-900 border-slate-800 text-slate-100 focus:border-indigo-500'
          }`}
        >
          {datasets.map(ds => (
            <option key={ds.id} value={ds.id}>
              {ds.name}
            </option>
          ))}
        </select>
        <div className="mt-1.5 text-[11px] opacity-70">
          {selectedDataset.rows} rows · {selectedDataset.columns} cols · {selectedDataset.suggested_task}
        </div>
      </div>

      {/* Constraints configuration (st.sidebar.slider/number_input) */}
      <div className="mb-6 space-y-3">
        <div className="text-[11px] font-bold uppercase tracking-wider opacity-60 font-mono flex items-center gap-1">
          <Settings className="w-3.5 h-3.5" />
          Pipeline Constraints
        </div>

        <div>
          <div className="flex justify-between text-xs mb-1">
            <span>Max Loop-Backs:</span>
            <span className="font-mono font-semibold">{constraints.max_loop_backs}</span>
          </div>
          <input
            type="range"
            min={1}
            max={5}
            value={constraints.max_loop_backs}
            onChange={(e) => onUpdateConstraints({ ...constraints, max_loop_backs: Number(e.target.value) })}
            className="w-full accent-[#ff4b4b] cursor-pointer"
          />
        </div>

        <div>
          <div className="flex justify-between text-xs mb-1">
            <span>Optuna CV Trials:</span>
            <span className="font-mono font-semibold">{constraints.max_tuning_trials}</span>
          </div>
          <input
            type="range"
            min={5}
            max={30}
            step={5}
            value={constraints.max_tuning_trials}
            onChange={(e) => onUpdateConstraints({ ...constraints, max_tuning_trials: Number(e.target.value) })}
            className="w-full accent-[#ff4b4b] cursor-pointer"
          />
        </div>

        <div>
          <div className="flex justify-between text-xs mb-1">
            <span>CV Folds:</span>
            <span className="font-mono font-semibold">{constraints.cv_folds}</span>
          </div>
          <input
            type="range"
            min={3}
            max={10}
            value={constraints.cv_folds}
            disabled
            title="Preview only: the pipeline's fold count (cv=3 in tuning_stage.py) is not a run setting yet."
            onChange={(e) => onUpdateConstraints({ ...constraints, cv_folds: Number(e.target.value) })}
            className="w-full accent-[#ff4b4b] cursor-not-allowed opacity-40"
          />
          <p className="mt-1 text-[10px] leading-snug opacity-60">
            Preview only — the pipeline uses a fixed 3-fold CV (tuning_stage.py).
          </p>
        </div>
      </div>

      {/* System Status Footer */}
      <div className="mt-auto pt-4 border-t border-slate-300/80 dark:border-slate-800 text-xs">
        <div className="flex items-center justify-between text-[11px]">
          <span className="flex items-center gap-1.5 opacity-80">
            <Cpu className="w-3.5 h-3.5" />
            FastAPI
          </span>
          <span className="flex items-center gap-1 font-mono text-emerald-500 font-semibold">
            {apiConnected ? (
              <>
                <CheckCircle className="w-3 h-3" /> Online
              </>
            ) : (
              <>
                <AlertCircle className="w-3 h-3 text-amber-500" /> Offline
              </>
            )}
          </span>
        </div>
        <div className="mt-1 text-[10px] opacity-60 font-mono">
          Optuna · scikit-learn · Postgres
        </div>
      </div>
    </aside>
  );
};
