import React from 'react';
import {
  BookOpen,
  Layers,
  Sliders,
  Play,
  RotateCcw,
  ShieldAlert,
  Database,
  CheckCircle2,
  Bug,
  Activity
} from 'lucide-react';

interface ArchitectureViewProps {
  uiTheme: 'streamlit' | 'modern';
}

export const ArchitectureView: React.FC<ArchitectureViewProps> = ({ uiTheme }) => {
  const isStreamlit = uiTheme === 'streamlit';

  return (
    <div className="space-y-6 max-w-6xl mx-auto w-full pb-12">
      {/* Header */}
      <div className="border-b pb-4 border-slate-200 dark:border-slate-800">
        <h2 className="text-2xl font-bold tracking-tight text-slate-900 dark:text-slate-100 flex items-center gap-2">
          <BookOpen className="w-6 h-6 text-[#ff4b4b] dark:text-indigo-400" />
          <span>Multi-Agent ML Pipeline Architecture</span>
        </h2>
        <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
          Full system technical specification, agent roles, deterministic loop-back state machine, and 26-table PostgreSQL relational schema.
        </p>
      </div>

      {/* Core Design Principle Card */}
      <div className={`p-4 rounded-xl border ${
        isStreamlit
          ? 'bg-amber-50/60 border-amber-200 text-amber-900'
          : 'bg-amber-950/20 border-amber-800/40 text-amber-200'
      }`}>
        <h3 className="text-sm font-bold flex items-center gap-2 mb-1.5">
          <span>💡 Core Design Philosophy</span>
        </h3>
        <p className="text-xs leading-relaxed">
          <strong>Deterministic code does all computation; LLMs only ever judge results that code already computed.</strong>{' '}
          No agent generates numbers, transforms data, or makes irreversible decisions — every LLM output is schema-constrained
          (Pydantic <code>response_format</code>) and every consequential decision (loop-back routing, whether to halt) is deterministic
          Python reading that judgment, never the judgment controlling flow directly.
        </p>
      </div>

      {/* The 5 Agents Table */}
      <div className={`p-5 rounded-xl border ${
        isStreamlit
          ? 'bg-white border-slate-200 shadow-xs'
          : 'bg-slate-900 border-slate-800'
      }`}>
        <h3 className="text-sm font-bold text-slate-900 dark:text-slate-100 mb-3 flex items-center gap-2">
          <Layers className="w-4 h-4 text-indigo-500" />
          The Five Autonomous Agents
        </h3>

        <div className="overflow-x-auto border rounded-lg border-slate-200 dark:border-slate-800 text-xs font-mono">
          <table className="w-full text-left">
            <thead className="bg-slate-50 dark:bg-slate-950 text-slate-500 text-[11px] border-b border-slate-200 dark:border-slate-800">
              <tr>
                <th className="p-2.5">Agent</th>
                <th className="p-2.5">Deterministic Tool</th>
                <th className="p-2.5">LLM Supervision Job</th>
                <th className="p-2.5">Key Judgment Field</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 dark:divide-slate-800/60 text-[11px]">
              <tr>
                <td className="p-2.5 font-bold text-indigo-600 dark:text-indigo-400">Requirement Agent</td>
                <td className="p-2.5">peek_dataset_tool</td>
                <td className="p-2.5 font-sans">Extract ProblemSpec grounded in real column names</td>
                <td className="p-2.5">unclear_or_missing</td>
              </tr>
              <tr>
                <td className="p-2.5 font-bold text-indigo-600 dark:text-indigo-400">Data Agent</td>
                <td className="p-2.5">profile_dataset_tool &rarr; clean_and_profile()</td>
                <td className="p-2.5 font-sans">Judge whether cleaned data is ready for feature engineering</td>
                <td className="p-2.5">proceed: bool (Blocking)</td>
              </tr>
              <tr>
                <td className="p-2.5 font-bold text-indigo-600 dark:text-indigo-400">Feature Agent</td>
                <td className="p-2.5">engineer_features_tool &rarr; engineer_features()</td>
                <td className="p-2.5 font-sans">
                  Judge feature quality. <strong>leakage_warnings is the single highest-priority signal</strong>
                </td>
                <td className="p-2.5 text-amber-500 font-bold">proceed: bool (Blocking)</td>
              </tr>
              <tr>
                <td className="p-2.5 font-bold text-indigo-600 dark:text-indigo-400">Tuning Agent</td>
                <td className="p-2.5">tune_model_tool &rarr; Optuna RandomForest</td>
                <td className="p-2.5 font-sans">Judge CV search trustworthiness; detects real plateau convergence</td>
                <td className="p-2.5 text-slate-400">proceed: bool (Advisory only)</td>
              </tr>
              <tr>
                <td className="p-2.5 font-bold text-indigo-600 dark:text-indigo-400">Training Agent</td>
                <td className="p-2.5">train_and_evaluate_tool &rarr; fit &amp; test</td>
                <td className="p-2.5 font-sans">Sanity-check rule failure analysis recommendation against real metrics</td>
                <td className="p-2.5 text-slate-400">agrees_with_rule_based_recommendation (Advisory)</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Orchestrator & Loop-Back Rules */}
      <div className={`p-5 rounded-xl border ${
        isStreamlit
          ? 'bg-white border-slate-200 shadow-xs'
          : 'bg-slate-900 border-slate-800'
      }`}>
        <h3 className="text-sm font-bold text-slate-900 dark:text-slate-100 mb-3 flex items-center gap-2">
          <RotateCcw className="w-4 h-4 text-amber-500" />
          Deterministic Orchestration &amp; Loop-Back State Machine
        </h3>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs">
          <div className="p-3 rounded-lg bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800">
            <h4 className="font-bold text-slate-800 dark:text-slate-200 mb-1">Resume-From Tracking</h4>
            <p className="text-slate-600 dark:text-slate-400 text-[11px] leading-relaxed">
              On failure, only re-invokes the stage actually implicated: <code>expand_hyperparam_search</code> skips Feature Agent entirely,
              while <code>revisit_features</code> re-runs from Feature Agent.
            </p>
          </div>

          <div className="p-3 rounded-lg bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800">
            <h4 className="font-bold text-slate-800 dark:text-slate-200 mb-1">Strict Halt Rules</h4>
            <p className="text-slate-600 dark:text-slate-400 text-[11px] leading-relaxed">
              <code>proceed=False</code> from Data or Feature halts immediately. <code>proceed=False</code> from Tuning does
              <strong> not</strong> halt because non-convergence doesn't reliably predict test failure.
            </p>
          </div>

          <div className="p-3 rounded-lg bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800">
            <h4 className="font-bold text-slate-800 dark:text-slate-200 mb-1">Deterministic Routing</h4>
            <p className="text-slate-600 dark:text-slate-400 text-[11px] leading-relaxed">
              Routing always reads <code>EvaluationReport.failure_analysis.recommended_next_step</code>, never LLM text. Retries are capped by{' '}
              <code>spec.constraints.max_loop_backs</code> (default 3).
            </p>
          </div>
        </div>
      </div>

      {/* Relational Database 26-table Architecture */}
      <div className={`p-5 rounded-xl border ${
        isStreamlit
          ? 'bg-white border-slate-200 shadow-xs'
          : 'bg-slate-900 border-slate-800'
      }`}>
        <h3 className="text-sm font-bold text-slate-900 dark:text-slate-100 mb-2 flex items-center gap-2">
          <Database className="w-4 h-4 text-emerald-500" />
          PostgreSQL Fully Relational Decomposition (26 Tables)
        </h3>
        <p className="text-xs text-slate-500 dark:text-slate-400 mb-3 leading-relaxed">
          Full relational decomposition (Option A) without JSONB shortcuts. Every nested Pydantic list/dict becomes its own child table with a foreign key back to its parent.
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-4 gap-2 font-mono text-[11px]">
          <div className="p-2 rounded bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800">
            <span className="text-slate-400 block text-[9px]">ROOT TABLE</span>
            <span className="font-bold text-slate-700 dark:text-slate-200">runs</span>
          </div>
          <div className="p-2 rounded bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800">
            <span className="text-slate-400 block text-[9px]">STAGE REPORTS (1:1)</span>
            <span className="text-slate-700 dark:text-slate-200 truncate block">problem_specs, data_profiles...</span>
          </div>
          <div className="p-2 rounded bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800">
            <span className="text-slate-400 block text-[9px]">COLLECTIONS (1:N)</span>
            <span className="text-slate-700 dark:text-slate-200 truncate block">column_schemas, trial_records...</span>
          </div>
          <div className="p-2 rounded bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800">
            <span className="text-slate-400 block text-[9px]">APPEND-ONLY LOGS</span>
            <span className="text-slate-700 dark:text-slate-200 truncate block">history_entries, trace_entries</span>
          </div>
        </div>
      </div>

      {/* Bugs Caught & Fixed via Automated Testing */}
      <div className={`p-5 rounded-xl border ${
        isStreamlit
          ? 'bg-white border-slate-200 shadow-xs'
          : 'bg-slate-900 border-slate-800'
      }`}>
        <h3 className="text-sm font-bold text-slate-900 dark:text-slate-100 mb-2 flex items-center gap-2">
          <Bug className="w-4 h-4 text-red-500" />
          Critical Bugs Identified &amp; Resolved by Test Suite (pytest suite)
        </h3>
        <ul className="space-y-2 text-xs text-slate-600 dark:text-slate-300 list-disc pl-4 leading-relaxed">
          <li>
            <strong>DataProfile Column Schema Indentation:</strong> Previously only ever populated a single column entry due to an early indentation flaw; fixed to profile all N columns.
          </li>
          <li>
            <strong>Optuna Plateau Convergence Bug:</strong> <code>TuningResult.converged</code> was structurally always <code>False</code> because <code>optimize(n_trials=n)</code> always runs <code>n</code> trials, making 2 of 3 failure branches unreachable. Replaced with real plateau slope evaluation.
          </li>
          <li>
            <strong>Regression Metric Crash:</strong> <code>mean_squared_error(squared=False)</code> was deprecated/removed in newer scikit-learn; fixed to support continuous targets cleanly.
          </li>
          <li>
            <strong>Target Leakage Silent Drop Bug:</strong> Ensured <code>leakage_warnings</code> explicitly alert rather than silently pruning, honoring the strict user transparency directive.
          </li>
        </ul>
      </div>
    </div>
  );
};
