import React from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ReferenceLine
} from 'recharts';
import { TuningResult } from '../types';
import { CheckCircle2, TrendingUp } from 'lucide-react';

interface OptunaTuningChartProps {
  tuning: TuningResult;
  metricThreshold?: number;
  metricName?: string;
}

export const OptunaTuningChart: React.FC<OptunaTuningChartProps> = ({
  tuning,
  metricThreshold = 0.85,
  metricName = 'Score'
}) => {
  const data = tuning.search_history.map(t => ({
    trial: `T${t.trial}`,
    trialNum: t.trial,
    cv_score: t.cv_score,
    n_estimators: t.params?.n_estimators,
    max_depth: t.params?.max_depth
  }));

  const minScore = Math.min(...data.map(d => d.cv_score), metricThreshold) * 0.96;
  const maxScore = Math.max(...data.map(d => d.cv_score), metricThreshold) * 1.03;

  return (
    <div className="space-y-4">
      {/* Top summary metrics */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <div className="p-3 rounded-lg bg-slate-900 border border-slate-800">
          <div className="text-xs text-slate-400">Best CV Score</div>
          <div className="text-xl font-bold text-emerald-400 font-mono mt-0.5">
            {tuning.best_cv_score.toFixed(4)}
          </div>
          <div className="text-[11px] text-slate-500 mt-0.5">RandomForest 3-Fold</div>
        </div>

        <div className="p-3 rounded-lg bg-slate-900 border border-slate-800">
          <div className="text-xs text-slate-400">Completed Trials</div>
          <div className="text-xl font-bold text-slate-100 font-mono mt-0.5">
            {tuning.search_history.length}
          </div>
          <div className="text-[11px] text-slate-500 mt-0.5">Optuna TPE Sampler</div>
        </div>

        <div className="p-3 rounded-lg bg-slate-900 border border-slate-800">
          <div className="text-xs text-slate-400">Plateau Convergence</div>
          <div className="flex items-center gap-1.5 mt-1">
            <CheckCircle2 className={`w-4 h-4 ${tuning.converged ? 'text-emerald-400' : 'text-amber-400'}`} />
            <span className={`text-sm font-semibold ${tuning.converged ? 'text-emerald-300' : 'text-amber-300'}`}>
              {tuning.converged ? 'Converged' : 'Searching'}
            </span>
          </div>
          <div className="text-[11px] text-slate-500 mt-0.5">Plateau over last 20% of trials</div>
        </div>

        <div className="p-3 rounded-lg bg-slate-900 border border-slate-800">
          <div className="text-xs text-slate-400">LLM Judgment</div>
          <div className="text-sm font-semibold text-blue-300 mt-1 flex items-center gap-1">
            <TrendingUp className="w-3.5 h-3.5" />
            Proceed: {tuning.agent_judgment.proceed === null ? 'not logged' : tuning.agent_judgment.proceed ? 'True' : 'False'}
          </div>
          <div className="text-[11px] text-slate-500 mt-0.5">Advisory (Non-blocking)</div>
        </div>
      </div>

      {/* Chart Canvas */}
      <div className="p-4 rounded-xl bg-slate-900/90 border border-slate-800">
        <div className="flex items-center justify-between mb-3 text-xs">
          <span className="font-semibold text-slate-300 flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-indigo-500 inline-block" />
            Optuna Cross-Validation Score Progression Across Trials
          </span>
          <div className="flex items-center gap-4 text-slate-400 font-mono">
            <span className="flex items-center gap-1">
              <span className="w-3 h-0.5 bg-indigo-400 inline-block" /> CV Score
            </span>
            <span className="flex items-center gap-1">
              <span className="w-3 h-0.5 bg-amber-400 border-dashed inline-block" /> Target ({metricThreshold})
            </span>
          </div>
        </div>

        <div className="h-56 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 10, right: 20, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="trial" stroke="#64748b" tick={{ fontSize: 11 }} />
              <YAxis domain={[minScore, maxScore]} stroke="#64748b" tick={{ fontSize: 11 }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#0f172a',
                  borderColor: '#334155',
                  borderRadius: '8px',
                  color: '#f8fafc',
                  fontSize: '12px'
                }}
                formatter={(val: any) => [Number(val).toFixed(4), `${metricName} (CV)`]}
              />
              <ReferenceLine
                y={metricThreshold}
                stroke="#f59e0b"
                strokeDasharray="4 4"
                label={{
                  value: `Target: ${metricThreshold}`,
                  fill: '#f59e0b',
                  fontSize: 10,
                  position: 'insideTopRight'
                }}
              />
              <Line
                type="monotone"
                dataKey="cv_score"
                stroke="#6366f1"
                strokeWidth={2.5}
                dot={{ fill: '#818cf8', r: 4 }}
                activeDot={{ r: 6, fill: '#38bdf8' }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Best Parameters Grid */}
      <div className="p-3.5 rounded-lg bg-slate-950 border border-slate-800 text-xs">
        <div className="font-semibold text-slate-300 mb-2">Optimal RandomForest Hyperparameters Found:</div>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 font-mono">
          {Object.entries(tuning.best_params).map(([k, v]) => (
            <div key={k} className="p-2 rounded bg-slate-900 border border-slate-800">
              <span className="text-slate-500 block text-[10px]">{k}</span>
              <span className="text-slate-200 font-semibold">{String(v)}</span>
            </div>
          ))}
        </div>
        <div className="mt-2 text-[11px] text-slate-400 leading-relaxed border-t border-slate-800/80 pt-2">
          <strong>Convergence Note:</strong> {tuning.convergence_notes}
        </div>
      </div>
    </div>
  );
};
