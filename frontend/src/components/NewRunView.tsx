import React, { useEffect, useRef, useState } from 'react';
import { Sparkles, Play, Sliders, Database, Upload, Loader2, AlertCircle } from 'lucide-react';
import { DatasetSummary, PipelineConstraints, ProblemSpec, TaskType, SuccessMetric } from '../types';

const CLASSIFICATION_METRICS = ['f1', 'accuracy', 'precision', 'recall', 'roc_auc'];
const REGRESSION_METRICS = ['r2', 'rmse', 'mae'];
const COLLAPSED_ROWS = 7;

export interface Clarification {
  question_for_user: string;
  missing_fields: string[];
}

interface NewRunViewProps {
  dataset: DatasetSummary;
  constraints: PipelineConstraints;
  uiTheme: 'streamlit' | 'modern';
  peekLoading: boolean;
  peekError: string | null;
  /** Upload a .csv/.parquet to the backend and make it the active dataset. */
  onUploadFile: (file: File) => Promise<void>;
  /** Structured path: start a run from a fully formed ProblemSpec. */
  onLaunchStructured: (spec: ProblemSpec) => Promise<void>;
  /** Plain-English path: the backend's Requirement Agent builds the spec and starts the run.
   *  Resolves with a clarification request instead of starting when the text is ambiguous. */
  onLaunchContext: (context: string) => Promise<Clarification | null>;
}

export const NewRunView: React.FC<NewRunViewProps> = ({
  dataset,
  constraints,
  uiTheme,
  peekLoading,
  peekError,
  onUploadFile,
  onLaunchStructured,
  onLaunchContext
}) => {
  const isStreamlit = uiTheme === 'streamlit';
  const [inputMode, setInputMode] = useState<'plain_text' | 'structured'>('plain_text');
  const [promptText, setPromptText] = useState(dataset.sample_prompt);
  const [taskType, setTaskType] = useState<TaskType>(dataset.suggested_task);
  const [targetCol, setTargetCol] = useState<string>(dataset.suggested_target);
  const [metric, setMetric] = useState<SuccessMetric>(dataset.suggested_metric);
  const [threshold, setThreshold] = useState<number>(0.9);

  const [launching, setLaunching] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [clarification, setClarification] = useState<Clarification | null>(null);
  const [uploading, setUploading] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  // Re-seed the form when the active dataset (or its schema, once peeked) changes.
  useEffect(() => {
    setPromptText(dataset.sample_prompt);
    setTaskType(dataset.suggested_task);
    setTargetCol(dataset.suggested_target);
    setMetric(dataset.suggested_metric);
    setClarification(null);
    setError(null);
    setExpanded(false);
  }, [dataset.id, dataset.suggested_target]);

  const handleFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setError(null);
    try {
      await onUploadFile(file);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed.');
    } finally {
      setUploading(false);
      if (fileInput.current) fileInput.current.value = '';
    }
  };

  const handleStart = async () => {
    setLaunching(true);
    setError(null);
    setClarification(null);
    try {
      if (inputMode === 'plain_text') {
        setClarification(await onLaunchContext(promptText.trim()));
      } else {
        await onLaunchStructured({
          task_type: taskType,
          target_column: targetCol,
          success_metric: metric,
          metric_threshold: threshold,
          data_source: dataset.id,
          constraints: {
            max_loop_backs: constraints.max_loop_backs,
            max_tuning_trials: constraints.max_tuning_trials
          }
        });
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not start the run.');
    } finally {
      setLaunching(false);
    }
  };

  const schemaRows = expanded ? dataset.schema : dataset.schema.slice(0, COLLAPSED_ROWS);
  const canLaunch =
    !launching && (inputMode === 'plain_text' ? promptText.trim().length > 0 : targetCol.length > 0);

  return (
    <div className="space-y-6 max-w-6xl mx-auto w-full pb-10">
      {/* Streamlit style Markdown header */}
      <div className="border-b pb-4 border-slate-200 dark:border-slate-800">
        <h2 className="text-2xl font-bold tracking-tight text-slate-900 dark:text-slate-100 flex items-center gap-2">
          <span>🚀 Initialize New Pipeline Run</span>
        </h2>
        <p className="text-xs text-slate-500 dark:text-slate-400 mt-1 leading-relaxed">
          Provide your dataset and plain-English intent. The <strong>Requirement Agent</strong> inspects schema
          peeks (never raw rows) and builds a validated <code>ProblemSpec</code> for the autonomous orchestrator.
        </p>
      </div>

      {/* Dataset Peek Card */}
      <div className={`p-4 rounded-xl border transition-all ${
        isStreamlit
          ? 'bg-white border-slate-200 shadow-xs text-slate-800'
          : 'bg-slate-900 border-slate-800 text-slate-200'
      }`}>
        <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
          <div className="flex items-center gap-2">
            <Database className="w-4 h-4 text-[#ff4b4b] dark:text-indigo-400" />
            <h3 className="text-sm font-bold">
              Active Dataset Peek: {dataset.name}
            </h3>
          </div>
          {dataset.rows > 0 && (
            <span className="text-xs px-2.5 py-0.5 rounded-full bg-slate-100 dark:bg-slate-800 font-mono text-slate-600 dark:text-slate-300">
              {dataset.rows} rows · {dataset.columns} columns
            </span>
          )}
        </div>

        <p className="text-xs text-slate-500 dark:text-slate-400 mb-3">
          {dataset.description}
        </p>

        {/* Schema Peek Table (peek_dataset_tool output) */}
        <div className="overflow-x-auto border rounded-lg border-slate-200 dark:border-slate-800">
          {peekLoading ? (
            <div className="flex items-center justify-center gap-2 py-8 text-xs text-slate-400">
              <Loader2 className="w-4 h-4 animate-spin" /> Reading schema…
            </div>
          ) : peekError ? (
            <div className="flex items-start gap-2 p-4 text-xs text-red-500">
              <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" />
              <span>{peekError}</span>
            </div>
          ) : (
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-50 dark:bg-slate-950/80 font-mono text-[11px] text-slate-500 border-b border-slate-200 dark:border-slate-800">
                <tr>
                  <th className="p-2.5">Column Name</th>
                  <th className="p-2.5">Data Type</th>
                  <th className="p-2.5">Missing %</th>
                  <th className="p-2.5">Cardinality</th>
                  <th className="p-2.5">Sample Values</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 dark:divide-slate-800/60 font-mono text-[11px]">
                {schemaRows.map((col) => (
                  <tr key={col.name} className="hover:bg-slate-50/60 dark:hover:bg-slate-800/30">
                    <td className="p-2.5 font-semibold text-slate-800 dark:text-slate-200">
                      {col.name} {col.name === targetCol && (
                        <span className="ml-1 text-[10px] text-[#ff4b4b] dark:text-amber-400 font-bold">
                          [Target]
                        </span>
                      )}
                    </td>
                    <td className="p-2.5 text-slate-500">{col.dtype}</td>
                    <td className="p-2.5 text-slate-500">{col.missing_pct}%</td>
                    <td className="p-2.5 text-slate-500">{col.cardinality}</td>
                    <td className="p-2.5 text-slate-400 truncate max-w-xs">
                      {col.sample_values.join(', ')}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          {!peekLoading && !peekError && dataset.schema.length > COLLAPSED_ROWS && (
            <button
              onClick={() => setExpanded((v) => !v)}
              className="w-full py-2 text-[11px] font-medium border-t border-slate-200 dark:border-slate-800 text-[#ff4b4b] dark:text-indigo-400 hover:bg-slate-50 dark:hover:bg-slate-800/40 cursor-pointer"
            >
              {expanded ? 'Show fewer columns' : `Show all ${dataset.schema.length} columns`}
            </button>
          )}
        </div>

        {/* Custom Upload */}
        <div className="mt-3 pt-3 border-t border-slate-100 dark:border-slate-800 flex items-center justify-between text-xs">
          <span className="text-slate-500 dark:text-slate-400">
            Want to test your own data? Upload a .csv file:
          </span>
          <label className="cursor-pointer inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-slate-300 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 hover:bg-slate-100 text-slate-700 dark:text-slate-200 font-medium">
            {uploading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Upload className="w-3.5 h-3.5" />}
            <span>Upload CSV</span>
            <input
              ref={fileInput}
              type="file"
              accept=".csv,.parquet"
              onChange={handleFile}
              disabled={uploading}
              className="hidden"
            />
          </label>
        </div>
      </div>

      {/* Input Configuration Mode Tabs */}
      <div className={`p-5 rounded-xl border ${
        isStreamlit
          ? 'bg-white border-slate-200 shadow-xs'
          : 'bg-slate-900 border-slate-800'
      }`}>
        <div className="flex border-b border-slate-200 dark:border-slate-800 mb-4 gap-4">
          <button
            onClick={() => setInputMode('plain_text')}
            className={`pb-2.5 text-xs font-semibold flex items-center gap-2 border-b-2 transition-all cursor-pointer ${
              inputMode === 'plain_text'
                ? isStreamlit
                  ? 'border-[#ff4b4b] text-[#ff4b4b]'
                  : 'border-indigo-500 text-indigo-400'
                : 'border-transparent text-slate-500 hover:text-slate-800 dark:hover:text-slate-200'
            }`}
          >
            <Sparkles className="w-4 h-4" />
            Requirement Agent (Natural Language Context)
          </button>

          <button
            onClick={() => setInputMode('structured')}
            className={`pb-2.5 text-xs font-semibold flex items-center gap-2 border-b-2 transition-all cursor-pointer ${
              inputMode === 'structured'
                ? isStreamlit
                  ? 'border-[#ff4b4b] text-[#ff4b4b]'
                  : 'border-indigo-500 text-indigo-400'
                : 'border-transparent text-slate-500 hover:text-slate-800 dark:hover:text-slate-200'
            }`}
          >
            <Sliders className="w-4 h-4" />
            Structured ProblemSpec Form
          </button>
        </div>

        {/* Tab 1: Requirement Agent Free Text */}
        {inputMode === 'plain_text' && (
          <div className="space-y-4">
            <div>
              <label className="block text-xs font-semibold text-slate-700 dark:text-slate-300 mb-1.5">
                Describe your modeling problem in plain English (context):
              </label>
              <textarea
                rows={3}
                value={promptText}
                onChange={(e) => setPromptText(e.target.value)}
                placeholder="E.g. Predict whether a patient tumor is malignant with F1 >= 0.90..."
                className={`w-full p-3 text-xs rounded-lg border outline-hidden transition-all ${
                  isStreamlit
                    ? 'bg-slate-50 border-slate-300 focus:border-[#ff4b4b] focus:bg-white text-slate-900'
                    : 'bg-slate-950 border-slate-800 focus:border-indigo-500 text-slate-100'
                }`}
              />
            </div>
            <p className="text-[11px] text-slate-500">
              The Requirement Agent grounds your text against the real schema above and asks for clarification
              rather than guessing a column. Launching starts the run immediately — there is no separate preview step.
            </p>

            {clarification && (
              <div className="p-3.5 rounded-lg bg-amber-50 dark:bg-amber-950/20 border border-amber-200 dark:border-amber-800/40 text-xs">
                <div className="font-semibold text-amber-800 dark:text-amber-300 mb-1">
                  Needs clarification — not an error.
                </div>
                <p className="text-amber-700 dark:text-amber-400 text-[11px] leading-relaxed">
                  {clarification.question_for_user}
                </p>
                <p className="mt-1.5 font-mono text-[10px] text-amber-600/80">
                  missing: {clarification.missing_fields.join(', ')}
                </p>
              </div>
            )}
          </div>
        )}

        {/* Tab 2: Structured Parameter Form */}
        {inputMode === 'structured' && (
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs">
            <div>
              <label className="block font-semibold text-slate-700 dark:text-slate-300 mb-1">
                Task Type
              </label>
              <select
                value={taskType}
                onChange={(e) => {
                  const next = e.target.value as TaskType;
                  setTaskType(next);
                  setMetric(next === 'classification' ? 'f1' : 'r2');
                }}
                className={`w-full p-2.5 rounded-lg border outline-hidden ${
                  isStreamlit
                    ? 'bg-slate-50 border-slate-300'
                    : 'bg-slate-950 border-slate-800 text-slate-100'
                }`}
              >
                <option value="classification">classification</option>
                <option value="regression">regression</option>
              </select>
            </div>

            <div>
              <label className="block font-semibold text-slate-700 dark:text-slate-300 mb-1">
                Target Column
              </label>
              <select
                value={targetCol}
                onChange={(e) => setTargetCol(e.target.value)}
                className={`w-full p-2.5 rounded-lg border outline-hidden ${
                  isStreamlit
                    ? 'bg-slate-50 border-slate-300'
                    : 'bg-slate-950 border-slate-800 text-slate-100'
                }`}
              >
                {dataset.schema.map(c => (
                  <option key={c.name} value={c.name}>{c.name}</option>
                ))}
              </select>
            </div>

            <div>
              <label className="block font-semibold text-slate-700 dark:text-slate-300 mb-1">
                Success Metric
              </label>
              <select
                value={metric}
                onChange={(e) => setMetric(e.target.value)}
                className={`w-full p-2.5 rounded-lg border outline-hidden ${
                  isStreamlit
                    ? 'bg-slate-50 border-slate-300'
                    : 'bg-slate-950 border-slate-800 text-slate-100'
                }`}
              >
                {(taskType === 'classification' ? CLASSIFICATION_METRICS : REGRESSION_METRICS).map((m) => (
                  <option key={m} value={m}>{m}</option>
                ))}
              </select>
            </div>

            <div>
              <label className="block font-semibold text-slate-700 dark:text-slate-300 mb-1">
                Metric Target Threshold: <span className="font-mono">{threshold}</span>
              </label>
              <input
                type="number"
                step={0.01}
                value={threshold}
                onChange={(e) => setThreshold(Number(e.target.value))}
                className={`w-full p-2.5 rounded-lg border outline-hidden font-mono ${
                  isStreamlit
                    ? 'bg-slate-50 border-slate-300'
                    : 'bg-slate-950 border-slate-800 text-slate-100'
                }`}
              />
              <p className="mt-1 text-[10px] text-slate-500">
                For error metrics (rmse, mae) this is a maximum, not a minimum.
              </p>
            </div>
          </div>
        )}

        {error && (
          <div className="mt-4 flex items-start gap-2 rounded-lg border border-red-300/60 dark:border-red-500/30 bg-red-50 dark:bg-red-500/10 p-3 text-xs text-red-600 dark:text-red-300">
            <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {/* Launch Button */}
        <div className="mt-6 pt-4 border-t border-slate-200 dark:border-slate-800 flex justify-end">
          <button
            id="launch-pipeline-btn"
            onClick={handleStart}
            disabled={!canLaunch}
            className={`px-6 py-2.5 rounded-lg font-bold text-xs flex items-center gap-2 text-white shadow-md transition-all cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed ${
              isStreamlit
                ? 'bg-[#ff4b4b] hover:bg-[#e03a3a]'
                : 'bg-emerald-600 hover:bg-emerald-700'
            }`}
          >
            {launching ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
            <span>{launching ? 'Starting…' : 'Launch Autonomous Pipeline'}</span>
          </button>
        </div>
      </div>
    </div>
  );
};
