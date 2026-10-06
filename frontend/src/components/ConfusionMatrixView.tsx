import React from 'react';
import { EvaluationReport, ProblemSpec } from '../types';
import { CheckCircle2, AlertTriangle, ArrowRight, ShieldCheck } from 'lucide-react';

interface ConfusionMatrixViewProps {
  evaluation: EvaluationReport;
  spec: ProblemSpec;
}

export const ConfusionMatrixView: React.FC<ConfusionMatrixViewProps> = ({
  evaluation,
  spec
}) => {
  const isClassification = spec.task_type === 'classification';
  const matrix = evaluation.confusion_matrix;
  const isPass = evaluation.pass_fail === 'PASS';
  const targetMetric = spec.success_metric;
  const targetThreshold = spec.metric_threshold;
  const achievedScore = evaluation.test_metrics[targetMetric] ?? 0;

  return (
    <div className="space-y-4">
      {/* Test Result Banner */}
      <div
        className={`p-4 rounded-xl border flex flex-wrap items-center justify-between gap-3 ${
          isPass
            ? 'bg-emerald-950/20 border-emerald-500/40 text-emerald-200'
            : 'bg-amber-950/20 border-amber-500/40 text-amber-200'
        }`}
      >
        <div className="flex items-center gap-3">
          {isPass ? (
            <div className="p-2 rounded-full bg-emerald-500/20 text-emerald-400">
              <CheckCircle2 className="w-5 h-5" />
            </div>
          ) : (
            <div className="p-2 rounded-full bg-amber-500/20 text-amber-400">
              <AlertTriangle className="w-5 h-5" />
            </div>
          )}
          <div>
            <div className="text-sm font-semibold flex items-center gap-2">
              <span>{isPass ? 'Evaluation Criteria Met' : 'Evaluation Threshold Missed'}</span>
              <span
                className={`text-[11px] font-bold px-2 py-0.5 rounded uppercase tracking-wider ${
                  isPass ? 'bg-emerald-600 text-white' : 'bg-amber-600 text-white'
                }`}
              >
                {evaluation.pass_fail}
              </span>
            </div>
            <p className="text-xs opacity-90 mt-0.5">
              Achieved test {targetMetric.toUpperCase()}:{' '}
              <strong className="font-mono text-white text-sm">{achievedScore.toFixed(4)}</strong> (Target Threshold:{' '}
              <span className="font-mono text-white">{targetThreshold}</span>)
            </p>
          </div>
        </div>

        {/* Delta indicator */}
        <div className="text-right text-xs">
          <div className="text-slate-400">Delta vs Target</div>
          <div
            className={`font-mono font-bold text-sm ${
              achievedScore >= targetThreshold ? 'text-emerald-400' : 'text-amber-400'
            }`}
          >
            {achievedScore >= targetThreshold ? '+' : ''}
            {(achievedScore - targetThreshold).toFixed(4)}
          </div>
        </div>
      </div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-5 gap-2.5 text-xs">
        {Object.entries(evaluation.test_metrics).map(([key, val]) => (
          <div
            key={key}
            className={`p-3 rounded-lg border ${
              key === targetMetric
                ? 'bg-indigo-950/40 border-indigo-500/50 ring-1 ring-indigo-500/30'
                : 'bg-slate-900 border-slate-800'
            }`}
          >
            <div className="text-slate-400 uppercase font-mono text-[10px] tracking-wider">
              {key} {key === targetMetric && '★ Primary'}
            </div>
            <div className="text-lg font-bold font-mono text-slate-100 mt-0.5">
              {typeof val === 'number' ? val.toFixed(4) : val}
            </div>
          </div>
        ))}
      </div>

      {/* Confusion Matrix (Classification) OR Residuals (Regression) */}
      {isClassification && matrix ? (
        <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 text-xs">
          <div className="font-semibold text-slate-200 mb-3 flex items-center justify-between">
            <span>Test-Set Confusion Matrix</span>
            <span className="text-slate-500 font-normal text-[11px]">N = {matrix.flat().reduce((a, b) => a + b, 0)} test samples</span>
          </div>

          {matrix.length !== 2 ? (
            <div className="overflow-x-auto">
              <table className="mx-auto font-mono text-[11px] text-slate-200">
                <thead>
                  <tr>
                    <th className="p-1.5 text-slate-500">actual \ predicted</th>
                    {matrix.map((_, j) => (
                      <th key={j} className="p-1.5 text-slate-400">{evaluation.matrix_labels?.[j] ?? `Class ${j}`}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {matrix.map((row, i) => (
                    <tr key={i}>
                      <td className="p-1.5 text-slate-400">{evaluation.matrix_labels?.[i] ?? `Class ${i}`}</td>
                      {row.map((cell, j) => (
                        <td
                          key={j}
                          className={`p-2.5 text-center border border-slate-800 ${i === j ? 'bg-emerald-950/30 text-emerald-300' : 'text-red-300'}`}
                        >
                          {cell}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
          <div className="max-w-xs mx-auto">
            <div className="text-center text-[11px] text-slate-400 mb-1 font-mono">PREDICTED CLASS</div>
            <div className="grid grid-cols-[auto_1fr_1fr] gap-2 items-center">
              <div />
              <div className="text-center text-[10px] font-mono text-slate-400">Class 0 (Neg)</div>
              <div className="text-center text-[10px] font-mono text-slate-400">Class 1 (Pos)</div>

              <div className="text-right text-[10px] font-mono text-slate-400 pr-1">ACTUAL Class 0</div>
              <div className="p-4 rounded-lg bg-emerald-950/30 border border-emerald-500/40 text-center">
                <div className="text-lg font-bold text-emerald-300 font-mono">{matrix[0][0]}</div>
                <div className="text-[10px] text-emerald-500">True Neg (TN)</div>
              </div>
              <div className="p-4 rounded-lg bg-red-950/20 border border-red-500/30 text-center">
                <div className="text-lg font-bold text-red-300 font-mono">{matrix[0][1]}</div>
                <div className="text-[10px] text-red-400">False Pos (FP)</div>
              </div>

              <div className="text-right text-[10px] font-mono text-slate-400 pr-1">ACTUAL Class 1</div>
              <div className="p-4 rounded-lg bg-red-950/20 border border-red-500/30 text-center">
                <div className="text-lg font-bold text-red-300 font-mono">{matrix[1][0]}</div>
                <div className="text-[10px] text-red-400">False Neg (FN)</div>
              </div>
              <div className="p-4 rounded-lg bg-emerald-950/30 border border-emerald-500/40 text-center">
                <div className="text-lg font-bold text-emerald-300 font-mono">{matrix[1][1]}</div>
                <div className="text-[10px] text-emerald-500">True Pos (TP)</div>
              </div>
            </div>
          </div>
          )}
        </div>
      ) : evaluation.residual_summary ? (
        <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 text-xs">
          <div className="font-semibold text-slate-200 mb-2">Regression Residual Diagnostics</div>
          <div className="grid grid-cols-2 gap-3 font-mono">
            <div className="p-2.5 rounded bg-slate-950 border border-slate-800">
              <span className="text-slate-500 block text-[10px]">Mean Residual</span>
              <span className="text-slate-200 font-bold">{evaluation.residual_summary.mean.toFixed(2)}</span>
            </div>
            <div className="p-2.5 rounded bg-slate-950 border border-slate-800">
              <span className="text-slate-500 block text-[10px]">Std of Residuals</span>
              <span className="text-slate-200 font-bold">{evaluation.residual_summary.std.toFixed(2)}</span>
            </div>
          </div>
        </div>
      ) : null}

      {/* Failure Analysis & Rule-Based Loop-Back routing */}
      <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 text-xs space-y-2">
        <div className="flex items-center justify-between">
          <div className="font-semibold text-slate-200 flex items-center gap-1.5">
            <ShieldCheck className="w-4 h-4 text-indigo-400" />
            Deterministic Failure Analysis & Loop-Back Route
          </div>
          <span className="text-[11px] font-mono text-indigo-300 bg-indigo-950/60 px-2 py-0.5 rounded border border-indigo-800">
            Next: {evaluation.failure_analysis.recommended_next_step}
          </span>
        </div>

        <p className="text-slate-300 leading-relaxed">
          {evaluation.failure_analysis.summary}
        </p>

        <div className="pt-2 border-t border-slate-800/80 text-[11px] text-slate-400 flex items-center justify-between">
          <span>
            <strong>Training Agent LLM Sanity Check:</strong>{' '}
            verdict is not persisted by the backend
          </span>
          <span className="text-slate-500">Advisory Only (Never affects routing directly)</span>
        </div>
      </div>
    </div>
  );
};
