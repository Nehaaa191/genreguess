import React from 'react';
import { RotateCcw, Trophy, Activity } from 'lucide-react';

export default function ResultsDisplay({ result, onReset }) {
  if (!result) return null;

  // Format probabilities for display, sorted descending
  const sortedProbs = Object.entries(result.probabilities)
    .sort(([, a], [, b]) => b - a)
    .map(([genre, prob]) => ({
      genre,
      prob: (prob * 100).toFixed(1),
      rawProb: prob
    }));

  return (
    <div className="w-full max-w-2xl mx-auto space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-700">
      
      {/* Primary Result Card */}
      <div className="bg-gradient-to-br from-brand-900/80 to-brand-950/80 rounded-2xl p-8 border border-brand-800 shadow-[0_0_40px_rgba(20,184,166,0.15)] text-center relative overflow-hidden">
        <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-brand-500 to-transparent opacity-50"></div>
        
        <div className="flex justify-center mb-4">
          <div className="w-16 h-16 rounded-full bg-brand-800/80 flex items-center justify-center border border-brand-600 shadow-lg">
            <Trophy className="w-8 h-8 text-brand-400" />
          </div>
        </div>
        
        <h3 className="text-brand-300 text-sm font-semibold tracking-wider uppercase mb-2">Predicted Genre</h3>
        <h1 className="text-5xl md:text-6xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-brand-100 to-brand-400 capitalize mb-4">
          {result.genre}
        </h1>
        
        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-brand-950 border border-brand-800">
          <Activity className="w-4 h-4 text-brand-500" />
          <span className="text-brand-200 font-medium">Confidence: {(result.confidence * 100).toFixed(1)}%</span>
        </div>
      </div>

      {/* Probabilities Chart */}
      <div className="bg-brand-900/40 rounded-xl p-6 border border-brand-800/50">
        <h3 className="text-brand-100 font-semibold mb-6 flex items-center gap-2">
          Top Probabilities
        </h3>
        
        <div className="space-y-4">
          {sortedProbs.slice(0, 5).map((item, index) => (
            <div key={item.genre} className="relative">
              <div className="flex justify-between text-sm mb-1">
                <span className={`capitalize font-medium ${index === 0 ? 'text-brand-200' : 'text-brand-400'}`}>
                  {item.genre}
                </span>
                <span className={index === 0 ? 'text-brand-200' : 'text-brand-400'}>
                  {item.prob}%
                </span>
              </div>
              <div className="h-2 w-full bg-brand-950 rounded-full overflow-hidden">
                <div 
                  className={`h-full rounded-full transition-all duration-1000 ease-out ${
                    index === 0 
                      ? 'bg-gradient-to-r from-brand-600 to-brand-400 shadow-[0_0_10px_rgba(20,184,166,0.5)]' 
                      : 'bg-brand-700'
                  }`}
                  style={{ width: `${item.prob}%`, transitionDelay: `${index * 100}ms` }}
                ></div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Reset Button */}
      <div className="flex justify-center mt-8">
        <button
          onClick={onReset}
          className="flex items-center gap-2 px-6 py-3 rounded-xl bg-brand-900 border border-brand-700 text-brand-300 hover:text-brand-100 hover:bg-brand-800 transition-colors font-medium shadow-lg"
        >
          <RotateCcw className="w-5 h-5" />
          Try Another Song
        </button>
      </div>

    </div>
  );
}
