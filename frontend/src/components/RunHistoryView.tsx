import React, { useState } from 'react';
import { Clock, Eye, CheckCircle2, AlertCircle, RotateCcw, Filter } from 'lucide-react';
import { PipelineRun } from '../types';

interface RunHistoryViewProps {
  runs: PipelineRun[];
  onSelectRun: (run: PipelineRun) => void;
  uiTheme: 'streamlit' | 'modern';
}

export const RunHistoryView: React.FC<RunHistoryViewProps> = ({
  runs,
  onSelectRun,
  uiTheme
}) => {
  const isStreamlit = uiTheme === 'streamlit';
  const [filterType, setFilterType] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState('');

  const filteredRuns = runs.filter(r => {
    if (filterType !== 'all' && r.problem_spec.task_type !== filterType) return false;
    if (searchQuery && !r.run_id.toLowerCase().includes(searchQuery.toLowerCase()) && !r.problem_spec.data_source.toLowerCase().includes(searchQuery.toLowerCase())) {
      return false;
    }
    return true;
  });

  return (
    <div className="space-y-6 max-w-6xl mx-auto w-full pb-10">
      {/* Streamlit Markdown header */}
      <div className="border-b pb-4 border-slate-200 dark:border-slate-800 flex flex-wrap items-end justify-between gap-3">
        <div className="min-w-0">
          <h2 className="text-2xl font-bold tracking-tight text-slate-900 dark:text-slate-100 flex items-center gap-2">
            <Clock className="w-6 h-6 text-[#ff4b4b] dark:text-indigo-400" />
            <span>Run History &amp; Execution Archive</span>
          </h2>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
            Persisted execution sessions matching the 26-table PostgreSQL schema. Click any run to inspect detailed agent judgments.
          </p>
        </div>

        {/* Filter controls */}
        <div className="flex shrink-0 items-center gap-2 text-xs">
          <div className="flex items-center gap-1 bg-slate-100 dark:bg-slate-800 p-1 rounded-lg border border-slate-200 dark:border-slate-700">
            <Filter className="w-3.5 h-3.5 ml-1 text-slate-400" />
            <select
              value={filterType}
              onChange={(e) => setFilterType(e.target.value)}
              className="bg-transparent text-xs p-1 outline-hidden"
            >
              <option value="all">All Tasks</option>
              <option value="classification">Classification</option>
              <option value="regression">Regression</option>
            </select>
          </div>

          <input
            type="text"
            placeholder="Search run ID or dataset..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className={`p-1.5 px-3 rounded-lg border text-xs outline-hidden ${
              isStreamlit
                ? 'bg-white border-slate-300 text-slate-900'
                : 'bg-slate-900 border-slate-800 text-slate-100'
            }`}
          />
        </div>
      </div>

      {filteredRuns.length === 0 ? (
        <div className="text-center py-16 text-slate-400 text-xs">
          No matching runs found in database. Launch your first run from the <strong>New Run</strong> tab!
        </div>
      ) : (
        <div className={`rounded-xl border overflow-hidden ${
          isStreamlit
            ? 'bg-white border-slate-200 shadow-xs'
            : 'bg-slate-900 border-slate-800'
        }`}>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs font-mono">
              <thead className="bg-slate-50 dark:bg-slate-950/80 border-b border-slate-200 dark:border-slate-800 text-slate-500 text-[11px]">
                <tr>
                  <th className="p-3">Run ID</th>
                  <th className="p-3">Dataset</th>
                  <th className="p-3">Task &amp; Target</th>
                  <th className="p-3">Loops</th>
                  <th className="p-3">Metric &amp; Score</th>
                  <th className="p-3">Pass/Fail</th>
                  <th className="p-3">Cost</th>
                  <th className="p-3 text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 dark:divide-slate-800/60 text-[11px]">
                {filteredRuns.map((r) => {
                  const metricName = r.problem_spec.success_metric;
                  const score = r.evaluation_report?.test_metrics?.[metricName];
                  const passFail = r.evaluation_report?.pass_fail;

                  return (
                    <tr key={r.run_id} className="hover:bg-slate-50/70 dark:hover:bg-slate-800/40 transition-colors">
                      <td className="p-3 font-bold text-slate-800 dark:text-slate-200 whitespace-nowrap" title={r.run_id}>
                        {r.run_id.slice(0, 8)}
                      </td>
                      <td
                        className="p-3 font-sans text-slate-600 dark:text-slate-300 truncate max-w-[180px]"
                        title={r.problem_spec.data_source}
                      >
                        {r.problem_spec.data_source.split(/[\\/]/).pop()}
                      </td>
                      <td
                        className="p-3 text-slate-500 truncate max-w-[220px]"
                        title={`${r.problem_spec.task_type}: ${r.problem_spec.target_column}`}
                      >
                        <span className="capitalize">{r.problem_spec.task_type}</span>: {r.problem_spec.target_column}
                      </td>
                      <td className="p-3">
                        {r.loop_count > 0 ? (
                          <span className="flex items-center gap-1 text-amber-500 font-semibold">
                            <RotateCcw className="w-3 h-3" />
                            {r.loop_count}
                          </span>
                        ) : (
                          <span className="text-slate-400">0</span>
                        )}
                      </td>
                      <td className="p-3 whitespace-nowrap">
                        <span className="uppercase text-slate-400 text-[10px] mr-1">{metricName}:</span>
                        <strong className="text-slate-800 dark:text-slate-200">
                          {score !== undefined ? score.toFixed(4) : '—'}
                        </strong>
                      </td>
                      <td className="p-3">
                        {passFail === 'PASS' ? (
                          <span className="px-2 py-0.5 rounded-full bg-emerald-100 text-emerald-700 dark:bg-emerald-950/60 dark:text-emerald-300 font-bold text-[10px]">
                            PASS
                          </span>
                        ) : passFail === 'FAIL' ? (
                          <span className="px-2 py-0.5 rounded-full bg-amber-100 text-amber-700 dark:bg-amber-950/60 dark:text-amber-300 font-bold text-[10px]">
                            FAIL
                          </span>
                        ) : (
                          <span className="px-2 py-0.5 rounded-full bg-blue-100 text-blue-700 dark:bg-blue-950/60 dark:text-blue-300 font-bold text-[10px]">
                            {r.status}
                          </span>
                        )}
                      </td>
                      <td className="p-3 text-emerald-600 dark:text-emerald-400 font-bold whitespace-nowrap">
                        ${r.total_cost_usd.toFixed(5)}
                      </td>
                      <td className="p-3 text-right">
                        <button
                          onClick={() => onSelectRun(r)}
                          className="px-2.5 py-1 rounded bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700 text-slate-800 dark:text-slate-200 text-xs font-semibold inline-flex items-center gap-1 whitespace-nowrap transition-all cursor-pointer"
                        >
                          <Eye className="w-3 h-3" />
                          <span>View Run</span>
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};
