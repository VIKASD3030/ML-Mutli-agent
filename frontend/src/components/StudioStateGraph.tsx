import React from 'react';
import { Sparkles, Layers, Sliders, Play, RotateCcw, Activity } from 'lucide-react';
import { PipelineRun, PipelineStage } from '../types';

interface StudioStateGraphProps {
  run: PipelineRun | null;
  onSelectStage?: (stage: string) => void;
  selectedStage?: string;
}

export const StudioStateGraph: React.FC<StudioStateGraphProps> = ({
  run,
  onSelectStage,
  selectedStage
}) => {
  const activeStage = run?.active_stage || 'idle';
  const status = run?.status || 'idle';
  const loopCount = run?.loop_count || 0;
  const isLooping = status === 'looping_back';

  const stages: {
    id: PipelineStage;
    label: string;
    agent: string;
    icon: React.ReactNode;
    desc: string;
  }[] = [
    {
      id: 'data',
      label: 'Data Agent',
      agent: 'clean_and_profile()',
      icon: <Layers className="w-4 h-4" />,
      desc: 'Imputation, outliers, class balance'
    },
    {
      id: 'feature',
      label: 'Feature Agent',
      agent: 'engineer_features()',
      icon: <Sliders className="w-4 h-4" />,
      desc: 'Leakage guard, encoding, interactions'
    },
    {
      id: 'tuning',
      label: 'Tuning Agent',
      agent: 'tune_hyperparameters()',
      icon: <Activity className="w-4 h-4" />,
      desc: 'Optuna CV search & plateau check'
    },
    {
      id: 'training',
      label: 'Training Agent',
      agent: 'train_and_evaluate()',
      icon: <Play className="w-4 h-4" />,
      desc: 'Final test fit & rule failure analysis'
    }
  ];

  const getStageStatus = (stageId: PipelineStage) => {
    if (!run) return 'idle';
    if (run.active_stage === stageId) return 'active';
    
    // Check if stage has completed reports
    if (stageId === 'data' && run.data_profile) return 'completed';
    if (stageId === 'feature' && run.eda_report) return 'completed';
    if (stageId === 'tuning' && run.tuning_result) return 'completed';
    if (stageId === 'training' && run.evaluation_report) {
      return run.evaluation_report.pass_fail === 'PASS' ? 'completed' : 'failed_eval';
    }
    return 'idle';
  };

  return (
    <div id="studio-state-graph" className="bg-slate-900 border border-slate-800 rounded-xl p-5 text-slate-100 shadow-md">
      <div className="flex flex-wrap items-center justify-between gap-2 mb-4 pb-3 border-b border-slate-800">
        <div>
          <div className="flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-emerald-400" />
            <h3 className="text-sm font-semibold tracking-wide text-slate-200">
              Orchestrator State Machine & Deterministic Routing
            </h3>
          </div>
          <p className="text-xs text-slate-400 mt-0.5">
            Sequential flow with automated loop-back failure recovery (Max loops: {run?.max_loop_backs ?? 3})
          </p>
        </div>

        {/* State badge */}
        <div className="flex items-center gap-2 text-xs">
          {isLooping && (
            <span className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-amber-500/20 text-amber-300 border border-amber-500/40 animate-pulse font-medium">
              <RotateCcw className="w-3.5 h-3.5 animate-spin" />
              Loop-Back #{loopCount} (Routing to: {run?.resume_from || 'feature'})
            </span>
          )}
          {status === 'running' && !isLooping && (
            <span className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-blue-500/20 text-blue-300 border border-blue-500/40">
              <span className="w-2 h-2 rounded-full bg-blue-400 animate-ping" />
              Active Stage: {activeStage.toUpperCase()}
            </span>
          )}
          {status === 'completed' && (
            <span className="px-2.5 py-1 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 font-medium">
              Pipeline Succeeded
            </span>
          )}
        </div>
      </div>

      {/* State Machine Nodes */}
      <div className="relative pt-2 pb-6">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 relative z-10">
          {stages.map((st, idx) => {
            const nodeStatus = getStageStatus(st.id);
            const isSelected = selectedStage === st.id;

            let borderClass = 'border-slate-800 bg-slate-950/60 text-slate-400';
            let badgeBg = 'bg-slate-800 text-slate-400';

            if (nodeStatus === 'active') {
              borderClass = 'border-blue-500 bg-blue-950/40 text-blue-100 ring-2 ring-blue-500/30 shadow-lg shadow-blue-500/10';
              badgeBg = 'bg-blue-600 text-white animate-pulse';
            } else if (nodeStatus === 'completed') {
              borderClass = 'border-emerald-500/50 bg-emerald-950/20 text-emerald-100';
              badgeBg = 'bg-emerald-600 text-white';
            } else if (nodeStatus === 'failed_eval') {
              borderClass = 'border-amber-500/50 bg-amber-950/20 text-amber-100';
              badgeBg = 'bg-amber-600 text-white';
            }

            return (
              <button
                key={st.id}
                id={`state-node-${st.id}`}
                onClick={() => onSelectStage && onSelectStage(st.id)}
                className={`text-left p-3.5 rounded-lg border transition-all cursor-pointer relative ${borderClass} ${
                  isSelected ? 'ring-2 ring-indigo-400' : 'hover:border-slate-600'
                }`}
              >
                <div className="flex items-center justify-between mb-2">
                  <span className={`text-[10px] font-mono px-1.5 py-0.5 rounded uppercase font-bold tracking-wider ${badgeBg}`}>
                    Stage 0{idx + 1}
                  </span>
                  <div className="p-1 rounded bg-slate-800/80 text-slate-300">
                    {st.icon}
                  </div>
                </div>

                <div className="font-semibold text-xs tracking-tight text-white flex items-center gap-1.5">
                  {st.label}
                </div>
                <div className="text-[11px] font-mono text-slate-400 mt-0.5 truncate">
                  {st.agent}
                </div>
                <div className="text-[11px] text-slate-400 mt-1.5 line-clamp-2 leading-relaxed">
                  {st.desc}
                </div>
              </button>
            );
          })}
        </div>

        {/* Loop Back Visual indicator */}
        {loopCount > 0 && (
          <div className="mt-3 p-2.5 rounded-lg bg-amber-950/30 border border-amber-800/40 flex items-center justify-between text-xs text-amber-300">
            <div className="flex items-center gap-2">
              <RotateCcw className="w-4 h-4 text-amber-400 shrink-0" />
              <span>
                <strong>Loop-Back State Active:</strong> Training Agent recommended{' '}
                <code className="bg-amber-900/50 px-1 py-0.5 rounded text-amber-200">
                  {run?.resume_from || 'revisit_features'}
                </code>
                . Orchestrator routed directly to Stage 02 without recomputing Data Profile.
              </span>
            </div>
            <span className="font-mono text-amber-400 font-semibold shrink-0 ml-2">
              Loops: {loopCount} / {run?.max_loop_backs || 3}
            </span>
          </div>
        )}
      </div>
    </div>
  );
};
