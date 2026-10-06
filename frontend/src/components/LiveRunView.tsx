import React, { useState } from 'react';
import {
  Sparkles,
  Layers,
  Sliders,
  Play,
  RotateCcw,
  CheckCircle2,
  AlertTriangle,
  Activity,
  Terminal,
  Clock,
  Coins,
  ShieldAlert,
  FileCode,
  Check
} from 'lucide-react';
import { PipelineRun, PipelineStage } from '../types';
import { StudioStateGraph } from './StudioStateGraph';
import { OptunaTuningChart } from './OptunaTuningChart';
import { ConfusionMatrixView } from './ConfusionMatrixView';

interface LiveRunViewProps {
  run: PipelineRun | null;
  uiTheme: 'streamlit' | 'modern';
  onReRun?: () => void;
}

export const LiveRunView: React.FC<LiveRunViewProps> = ({
  run,
  uiTheme,
  onReRun
}) => {
  const isStreamlit = uiTheme === 'streamlit';
  const [activeTab, setActiveTab] = useState<'overview' | 'data' | 'feature' | 'tuning' | 'training' | 'trace'>('overview');
  const [copiedJson, setCopiedJson] = useState(false);

  if (!run) {
    return (
      <div className="text-center py-20 text-slate-500 max-w-md mx-auto">
        <Activity className="w-12 h-12 mx-auto mb-3 opacity-30 animate-pulse" />
        <h3 className="text-sm font-semibold text-slate-700 dark:text-slate-300">No Active Pipeline Run</h3>
        <p className="text-xs mt-1">
          Select a dataset and launch a new autonomous run from the <strong>New Run</strong> tab to monitor live progress.
        </p>
      </div>
    );
  }

  const status = run.status;
  const isRunning = status === 'running' || status === 'looping_back';
  const primaryMetric = run.problem_spec.success_metric;
  const targetThreshold = run.problem_spec.metric_threshold;
  const finalScore = run.evaluation_report?.test_metrics?.[primaryMetric];
  const delta = finalScore ? finalScore - targetThreshold : null;

  const handleCopyJson = () => {
    navigator.clipboard.writeText(JSON.stringify(run, null, 2));
    setCopiedJson(true);
    setTimeout(() => setCopiedJson(false), 2000);
  };

  return (
    <div className="space-y-5 max-w-6xl mx-auto w-full pb-12">
      {/* Top Status & Metrics Bar (Streamlit st.metric) */}
      <div className={`p-4 rounded-xl border ${
        isStreamlit
          ? 'bg-white border-slate-200 shadow-xs'
          : 'bg-slate-900 border-slate-800'
      }`}>
        <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-slate-100 dark:border-slate-800">
          <div>
            <div className="flex items-center gap-2">
              <span className="font-mono text-xs text-slate-400">Run ID:</span>
              <span className="font-mono text-xs font-bold text-slate-800 dark:text-slate-200 bg-slate-100 dark:bg-slate-800 px-2 py-0.5 rounded">
                {run.run_id}
              </span>
              <span className={`text-[11px] font-bold px-2.5 py-0.5 rounded-full uppercase tracking-wider ${
                status === 'completed'
                  ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-950/50 dark:text-emerald-300'
                  : status === 'looping_back'
                  ? 'bg-amber-100 text-amber-700 dark:bg-amber-950/50 dark:text-amber-300 animate-pulse'
                  : status === 'failed'
                  ? 'bg-red-100 text-red-700 dark:bg-red-950/50 dark:text-red-300'
                  : 'bg-blue-100 text-blue-700 dark:bg-blue-950/50 dark:text-blue-300 animate-pulse'
              }`}>
                {status}
              </span>
            </div>
            <div className="text-xs text-slate-500 dark:text-slate-400 mt-1">
              Dataset: <strong>{run.problem_spec.data_source}</strong> · Task: <strong>{run.problem_spec.task_type}</strong> (target: <code>{run.problem_spec.target_column}</code>)
            </div>
          </div>

          <div className="flex items-center gap-4 text-xs font-mono">
            <div className="text-right">
              <span className="text-slate-400 block text-[10px]">LOOP COUNT</span>
              <span className="font-bold text-slate-700 dark:text-slate-200">
                {run.loop_count} / {run.max_loop_backs}
              </span>
            </div>
            <div className="text-right">
              <span className="text-slate-400 block text-[10px]">EST. COST</span>
              <span className="font-bold text-emerald-600 dark:text-emerald-400 flex items-center gap-1">
                <Coins className="w-3 h-3" />
                ${run.total_cost_usd.toFixed(5)}
              </span>
            </div>
            <div className="text-right">
              <span className="text-slate-400 block text-[10px]">TOTAL TOKENS</span>
              <span className="font-bold text-slate-700 dark:text-slate-200">
                {run.total_tokens.toLocaleString()}
              </span>
            </div>
          </div>
        </div>

        {/* Streamlit Metric delta cards */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-3">
          <div className="p-3 rounded-lg bg-slate-50 dark:bg-slate-950 border border-slate-100 dark:border-slate-800/80">
            <div className="text-[11px] text-slate-500 uppercase font-mono">Target Metric</div>
            <div className="text-base font-bold font-mono text-slate-800 dark:text-slate-100 mt-0.5 uppercase">
              {primaryMetric}
            </div>
            <div className="text-[10px] text-slate-400 mt-0.5">Threshold: {targetThreshold}</div>
          </div>

          <div className="p-3 rounded-lg bg-slate-50 dark:bg-slate-950 border border-slate-100 dark:border-slate-800/80">
            <div className="text-[11px] text-slate-500 uppercase font-mono">Achieved Score</div>
            <div className="text-base font-bold font-mono text-slate-800 dark:text-slate-100 mt-0.5">
              {finalScore !== undefined ? finalScore.toFixed(4) : isRunning ? 'Evaluating...' : 'N/A'}
            </div>
            {delta !== null && (
              <div className={`text-[11px] font-mono font-semibold ${delta >= 0 ? 'text-emerald-500' : 'text-amber-500'}`}>
                {delta >= 0 ? `+${delta.toFixed(4)}` : delta.toFixed(4)} vs threshold
              </div>
            )}
          </div>

          <div className="p-3 rounded-lg bg-slate-50 dark:bg-slate-950 border border-slate-100 dark:border-slate-800/80">
            <div className="text-[11px] text-slate-500 uppercase font-mono">Best Optuna CV</div>
            <div className="text-base font-bold font-mono text-slate-800 dark:text-slate-100 mt-0.5">
              {run.tuning_result ? run.tuning_result.best_cv_score.toFixed(4) : isRunning ? 'Searching...' : 'N/A'}
            </div>
            <div className="text-[10px] text-slate-400 mt-0.5">RandomForest 3-Fold</div>
          </div>

          <div className="p-3 rounded-lg bg-slate-50 dark:bg-slate-950 border border-slate-100 dark:border-slate-800/80">
            <div className="text-[11px] text-slate-500 uppercase font-mono">Pipeline Status</div>
            <div className="text-sm font-bold text-slate-800 dark:text-slate-100 mt-0.5 flex items-center gap-1.5">
              {status === 'completed' && <CheckCircle2 className="w-4 h-4 text-emerald-500" />}
              {status === 'looping_back' && <RotateCcw className="w-4 h-4 text-amber-500 animate-spin" />}
              {status === 'running' && <Activity className="w-4 h-4 text-blue-500 animate-pulse" />}
              <span className="capitalize">{status}</span>
            </div>
            <div className="text-[10px] text-slate-400 mt-0.5">Stage: {run.active_stage}</div>
          </div>
        </div>
      </div>

      {/* State Machine Flow Graph */}
      <StudioStateGraph
        run={run}
        onSelectStage={(stage) => {
          if (stage === 'data') setActiveTab('data');
          else if (stage === 'feature') setActiveTab('feature');
          else if (stage === 'tuning') setActiveTab('tuning');
          else if (stage === 'training') setActiveTab('training');
        }}
        selectedStage={activeTab}
      />

      {/* Streamlit Live Status Expander (st.status) */}
      {isRunning && (
        <div className="p-3.5 rounded-xl border border-blue-200 dark:border-blue-900/50 bg-blue-50/50 dark:bg-blue-950/20 text-xs flex items-center justify-between">
          <div className="flex items-center gap-2 text-blue-800 dark:text-blue-300">
            <div className="w-3 h-3 rounded-full border-2 border-blue-600 border-t-transparent animate-spin" />
            <span className="font-semibold">Autonomous ML Execution in progress...</span>
            <span className="text-slate-500 dark:text-slate-400">
              Active step: <strong>{run.active_stage.toUpperCase()}</strong> (Deterministic code computing, LLM supervising)
            </span>
          </div>
          <span className="font-mono text-[11px] text-blue-700 dark:text-blue-400 bg-blue-100 dark:bg-blue-900/60 px-2 py-0.5 rounded">
            Polling /runs/{run.run_id}
          </span>
        </div>
      )}

      {/* Navigation Tabs (st.tabs) */}
      <div className={`p-4 rounded-xl border ${
        isStreamlit
          ? 'bg-white border-slate-200 shadow-xs'
          : 'bg-slate-900 border-slate-800'
      }`}>
        <div className="flex flex-wrap gap-2 border-b border-slate-200 dark:border-slate-800 pb-3 mb-4">
          <button
            onClick={() => setActiveTab('overview')}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
              activeTab === 'overview'
                ? isStreamlit
                  ? 'bg-[#ff4b4b] text-white shadow-xs'
                  : 'bg-indigo-600 text-white shadow-xs'
                : 'text-slate-500 hover:text-slate-900 dark:hover:text-slate-200'
            }`}
          >
            📋 Overview &amp; Logs
          </button>

          <button
            onClick={() => setActiveTab('data')}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-all cursor-pointer ${
              activeTab === 'data'
                ? isStreamlit
                  ? 'bg-[#ff4b4b] text-white shadow-xs'
                  : 'bg-indigo-600 text-white shadow-xs'
                : 'text-slate-500 hover:text-slate-900 dark:hover:text-slate-200'
            }`}
          >
            <Layers className="w-3.5 h-3.5" />
            <span>Data Profile</span>
            {run.data_profile && <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />}
          </button>

          <button
            onClick={() => setActiveTab('feature')}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-all cursor-pointer ${
              activeTab === 'feature'
                ? isStreamlit
                  ? 'bg-[#ff4b4b] text-white shadow-xs'
                  : 'bg-indigo-600 text-white shadow-xs'
                : 'text-slate-500 hover:text-slate-900 dark:hover:text-slate-200'
            }`}
          >
            <Sliders className="w-3.5 h-3.5" />
            <span>Feature EDA &amp; Leakage</span>
            {run.eda_report?.leakage_warnings?.length ? (
              <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
            ) : run.eda_report ? (
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
            ) : null}
          </button>

          <button
            onClick={() => setActiveTab('tuning')}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-all cursor-pointer ${
              activeTab === 'tuning'
                ? isStreamlit
                  ? 'bg-[#ff4b4b] text-white shadow-xs'
                  : 'bg-indigo-600 text-white shadow-xs'
                : 'text-slate-500 hover:text-slate-900 dark:hover:text-slate-200'
            }`}
          >
            <Activity className="w-3.5 h-3.5" />
            <span>Optuna Tuning</span>
            {run.tuning_result && <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />}
          </button>

          <button
            onClick={() => setActiveTab('training')}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-all cursor-pointer ${
              activeTab === 'training'
                ? isStreamlit
                  ? 'bg-[#ff4b4b] text-white shadow-xs'
                  : 'bg-indigo-600 text-white shadow-xs'
                : 'text-slate-500 hover:text-slate-900 dark:hover:text-slate-200'
            }`}
          >
            <Play className="w-3.5 h-3.5" />
            <span>Evaluation &amp; Matrix</span>
            {run.evaluation_report && <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />}
          </button>

          <button
            onClick={() => setActiveTab('trace')}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-all cursor-pointer ${
              activeTab === 'trace'
                ? isStreamlit
                  ? 'bg-[#ff4b4b] text-white shadow-xs'
                  : 'bg-indigo-600 text-white shadow-xs'
                : 'text-slate-500 hover:text-slate-900 dark:hover:text-slate-200'
            }`}
          >
            <Coins className="w-3.5 h-3.5" />
            <span>Trace &amp; Observability</span>
          </button>
        </div>

        {/* TAB 1: Overview & Event Log Stream */}
        {activeTab === 'overview' && (
          <div className="space-y-4 text-xs">
            <div className="p-3.5 rounded-lg bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800">
              <h4 className="font-semibold text-slate-800 dark:text-slate-200 mb-1">
                Problem Specification (Grounded in Schema)
              </h4>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 font-mono text-[11px] mt-2">
                <div className="p-2 rounded bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800">
                  <span className="text-slate-400 block text-[9px]">TASK</span>
                  <span className="font-bold">{run.problem_spec.task_type}</span>
                </div>
                <div className="p-2 rounded bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800">
                  <span className="text-slate-400 block text-[9px]">TARGET</span>
                  <span className="font-bold">{run.problem_spec.target_column}</span>
                </div>
                <div className="p-2 rounded bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800">
                  <span className="text-slate-400 block text-[9px]">METRIC GOAL</span>
                  <span className="font-bold">{run.problem_spec.success_metric} &ge; {run.problem_spec.metric_threshold}</span>
                </div>
                <div className="p-2 rounded bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800">
                  <span className="text-slate-400 block text-[9px]">MAX RETRIES</span>
                  <span className="font-bold">{run.problem_spec.constraints.max_loop_backs} Loops</span>
                </div>
              </div>
            </div>

            {/* Event Log Stream */}
            <div className="p-3.5 rounded-lg bg-slate-950 text-slate-200 border border-slate-800">
              <div className="flex items-center justify-between mb-2">
                <div className="font-semibold flex items-center gap-1.5 font-mono text-slate-300">
                  <Terminal className="w-3.5 h-3.5 text-emerald-400" />
                  Orchestrator Event Log Stream
                </div>
                <span className="text-[10px] text-slate-500 font-mono">
                  {run.history.length} records
                </span>
              </div>

              <div className="space-y-1.5 font-mono text-[11px] max-h-56 overflow-y-auto pr-1">
                {run.history.map((h, i) => (
                  <div key={i} className="flex items-start gap-2 py-0.5 border-b border-slate-900 last:border-0">
                    <span className="text-slate-500 shrink-0 text-[10px]">{h.timestamp}</span>
                    <span className={`px-1.5 py-0.2 rounded text-[10px] shrink-0 ${
                      h.type === 'success' ? 'bg-emerald-950 text-emerald-300 border border-emerald-800' :
                      h.type === 'loop_back' ? 'bg-amber-950 text-amber-300 border border-amber-800' :
                      h.type === 'error' ? 'bg-red-950 text-red-300 border border-red-800' :
                      'bg-slate-800 text-slate-400'
                    }`}>
                      {h.stage}
                    </span>
                    <span className="text-slate-300 break-words">{h.message}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* TAB 2: Data Profile */}
        {activeTab === 'data' && (
          <div className="space-y-4 text-xs">
            {!run.data_profile ? (
              <div className="text-center py-8 text-slate-400">Data Agent has not executed yet.</div>
            ) : (
              <>
                {/* Agent Judgment Card */}
                <div className="p-3.5 rounded-lg bg-emerald-50 dark:bg-emerald-950/20 border border-emerald-200 dark:border-emerald-800/40">
                  <div className="flex items-center justify-between font-semibold text-emerald-800 dark:text-emerald-300 mb-1">
                    <span className="flex items-center gap-1.5">
                      <CheckCircle2 className="w-4 h-4" />
                      Data Agent LLM Judgment (Schema-Constrained)
                    </span>
                    <span className="font-mono text-[11px]">
                      proceed={run.data_profile.agent_judgment.proceed === null ? 'not logged' : String(run.data_profile.agent_judgment.proceed)}
                    </span>
                  </div>
                  <p className="text-emerald-700 dark:text-emerald-400 text-[11px] leading-relaxed">
                    The LLM's verdict is advisory; the backend records only the proceed flag (see the event log for the full trail).
                  </p>
                </div>

                {/* Cleaning actions & Class Balance */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  <div className="p-3.5 rounded-lg bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800">
                    <div className="font-semibold text-slate-800 dark:text-slate-200 mb-2">
                      Cleaning Actions Taken (Deterministic Layer):
                    </div>
                    <ul className="space-y-1.5 list-disc pl-4 text-slate-600 dark:text-slate-300 text-[11px]">
                      {run.data_profile.cleaning_actions_taken.map((act, i) => (
                        <li key={i}>{act}</li>
                      ))}
                    </ul>
                  </div>

                  <div className="p-3.5 rounded-lg bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800">
                    <div className="font-semibold text-slate-800 dark:text-slate-200 mb-2">
                      Class Balance &amp; Target Integrity:
                    </div>
                    {run.data_profile.class_balance ? (
                      <div className="space-y-2">
                        {Object.entries(run.data_profile.class_balance).map(([k, v]) => (
                          <div key={k} className="p-2 rounded bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 flex justify-between font-mono">
                            <span>{k}</span>
                            <span className="font-bold text-slate-800 dark:text-slate-200">{v} samples</span>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="text-slate-400 text-[11px]">
                        Continuous target distribution verified for regression task. No missing labels.
                      </div>
                    )}
                  </div>
                </div>

                {/* Outlier Flags */}
                {run.data_profile.outlier_flags.length > 0 && (
                  <div className="p-3.5 rounded-lg bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800">
                    <div className="font-semibold text-slate-800 dark:text-slate-200 mb-2">
                      Outlier Flags via Interquartile Range (IQR):
                    </div>
                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 font-mono text-[11px]">
                      {run.data_profile.outlier_flags.map((o, idx) => (
                        <div key={idx} className="p-2 rounded bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800">
                          <span className="font-bold text-slate-700 dark:text-slate-200 block break-words">{o}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </>
            )}
          </div>
        )}

        {/* TAB 3: Feature EDA & Leakage */}
        {activeTab === 'feature' && (
          <div className="space-y-4 text-xs">
            {!run.eda_report ? (
              <div className="text-center py-8 text-slate-400">Feature Agent has not executed yet.</div>
            ) : (
              <>
                {/* Feature Leakage Shield Banner (CRITICAL PRIORITY IN SPEC) */}
                {run.eda_report.leakage_warnings.length > 0 ? (
                  <div className="p-4 rounded-xl bg-red-950/30 border-2 border-red-500/60 text-red-200">
                    <div className="flex items-center gap-2 font-bold text-sm text-red-400 mb-1">
                      <ShieldAlert className="w-5 h-5 text-red-400 animate-bounce" />
                      CRITICAL TARGET LEAKAGE DETECTED
                    </div>
                    <p className="text-xs text-red-300 leading-relaxed mb-3">
                      <strong>System Directive:</strong> <em>Leakage warnings are the single highest-priority signal.</em>{' '}
                      The Feature Agent never silently drops leakage columns; it raises explicit alarms to prevent synthetic test score inflation.
                    </p>
                    <div className="space-y-2">
                      {run.eda_report.leakage_warnings.map((lw, i) => (
                        <div key={i} className="p-3 rounded-lg bg-red-950/60 border border-red-800/80 text-[11px]">
                          <p className="text-red-200 font-sans text-xs">{lw}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : (
                  <div className="p-3.5 rounded-lg bg-emerald-50 dark:bg-emerald-950/20 border border-emerald-200 dark:border-emerald-800/40 text-emerald-800 dark:text-emerald-300 flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <CheckCircle2 className="w-4 h-4" />
                      <span><strong>Leakage Shield Passed:</strong> No target leakage detected across candidate features.</span>
                    </div>
                    <span className="font-mono text-[11px]">|corr| &lt; 0.98</span>
                  </div>
                )}

                {/* Engineered vs Dropped Features */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  <div className="p-3.5 rounded-lg bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800">
                    <div className="font-semibold text-slate-800 dark:text-slate-200 mb-2">
                      Engineered Features &amp; Transformations:
                    </div>
                    <div className="space-y-2">
                      {run.eda_report.engineered_features.map((ef, i) => (
                        <div key={i} className="p-2 rounded bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800">
                          <div className="font-mono font-bold text-indigo-600 dark:text-indigo-400">{ef.name}</div>
                          <p className="text-slate-600 dark:text-slate-300 text-[11px] mt-1">{ef.rationale}</p>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="p-3.5 rounded-lg bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800">
                    <div className="font-semibold text-slate-800 dark:text-slate-200 mb-2">
                      Dropped Features:
                    </div>
                    <div className="space-y-2">
                      {run.eda_report.dropped_features.map((df, i) => (
                        <div key={i} className="p-2 rounded bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800">
                          <div className="font-mono font-bold text-slate-700 dark:text-slate-300">{df.name}</div>
                          <p className="text-slate-600 dark:text-slate-400 text-[11px] mt-1">{df.rationale}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                {/* Correlation Summary */}
                <div className="p-3.5 rounded-lg bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800">
                  <div className="font-semibold text-slate-800 dark:text-slate-200 mb-2">
                    Top Feature Correlations With Target:
                  </div>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 font-mono text-[11px]">
                    {run.eda_report.correlation_summary.map((c, i) => (
                      <div key={i} className="p-2 rounded bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800">
                        <span className="text-slate-500 block truncate text-[10px]">{c.feature_a} vs {c.feature_b}</span>
                        <span className="font-bold text-slate-700 dark:text-slate-200 mt-0.5 block">{c.correlation.toFixed(2)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </>
            )}
          </div>
        )}

        {/* TAB 4: Tuning Result & Optuna Convergence */}
        {activeTab === 'tuning' && (
          <div className="space-y-4">
            {!run.tuning_result ? (
              <div className="text-center py-8 text-slate-400 text-xs">Tuning Agent has not executed yet.</div>
            ) : (
              <OptunaTuningChart
                tuning={run.tuning_result}
                metricThreshold={run.problem_spec.metric_threshold}
                metricName={run.problem_spec.success_metric.toUpperCase()}
              />
            )}
          </div>
        )}

        {/* TAB 5: Evaluation Report & Confusion Matrix */}
        {activeTab === 'training' && (
          <div className="space-y-4">
            {!run.evaluation_report ? (
              <div className="text-center py-8 text-slate-400 text-xs">Training Agent has not completed final evaluation yet.</div>
            ) : (
              <ConfusionMatrixView
                evaluation={run.evaluation_report}
                spec={run.problem_spec}
              />
            )}
          </div>
        )}

        {/* TAB 6: Trace & Observability */}
        {activeTab === 'trace' && (
          <div className="space-y-4 text-xs">
            <div className="p-3.5 rounded-lg bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800">
              <div className="flex items-center justify-between mb-3">
                <div className="font-semibold text-slate-800 dark:text-slate-200">
                  Per-Agent LLM Observability &amp; Cost Telemetry
                </div>
                <button
                  onClick={handleCopyJson}
                  className="px-2.5 py-1 rounded bg-slate-200 dark:bg-slate-800 hover:bg-slate-300 font-mono text-[11px] flex items-center gap-1 cursor-pointer"
                >
                  {copiedJson ? <Check className="w-3 h-3 text-emerald-500" /> : <FileCode className="w-3 h-3" />}
                  <span>{copiedJson ? 'Copied' : 'Copy Full Run JSON'}</span>
                </button>
              </div>

              <div className="overflow-x-auto border rounded-lg border-slate-200 dark:border-slate-800">
                <table className="w-full text-left font-mono text-[11px]">
                  <thead className="bg-slate-100 dark:bg-slate-900 border-b border-slate-200 dark:border-slate-800 text-slate-500">
                    <tr>
                      <th className="p-2.5">Agent</th>
                      <th className="p-2.5">Model</th>
                      <th className="p-2.5">Prompt Tk</th>
                      <th className="p-2.5">Comp Tk</th>
                      <th className="p-2.5">Latency</th>
                      <th className="p-2.5">Est. Cost (USD)</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100 dark:divide-slate-800/60">
                    {run.trace.map((t, idx) => (
                      <tr key={idx} className="hover:bg-slate-50 dark:hover:bg-slate-900/50">
                        <td className="p-2.5 font-bold text-indigo-600 dark:text-indigo-400">{t.agent}</td>
                        <td className="p-2.5 text-slate-600 dark:text-slate-300">{t.model}</td>
                        <td className="p-2.5 text-slate-500">{t.prompt_tokens}</td>
                        <td className="p-2.5 text-slate-500">{t.completion_tokens}</td>
                        <td className="p-2.5 text-slate-500">{t.latency_ms} ms</td>
                        <td className="p-2.5 font-bold text-emerald-600 dark:text-emerald-400">${t.estimated_cost_usd.toFixed(6)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Relational DB Schema Note */}
            <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 text-slate-400 text-[11px] leading-relaxed">
              <strong>PostgreSQL Relational Bridge (`db/sync.py`):</strong> This state matches the 26-table relational schema
              persisted at every agent stage through the <code>on_update</code> callback, supporting full crash recovery and idempotent replay.
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
