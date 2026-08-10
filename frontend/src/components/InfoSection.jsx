import React from 'react';
import { Database, Network, Cpu, Layout, Server, Music, Activity, BarChart2 } from 'lucide-react';

export default function InfoSection() {
  return (
    <div className="mt-12 grid grid-cols-1 md:grid-cols-2 gap-8 w-full max-w-5xl mx-auto px-4">
      
      {/* How it works */}
      <div className="bg-brand-900/50 rounded-xl p-6 border border-brand-800 shadow-xl">
        <h2 className="text-xl font-bold mb-6 flex items-center gap-2 text-brand-100">
          <Activity className="w-5 h-5 text-brand-500" />
          How it works
        </h2>
        
        <div className="relative border-l-2 border-brand-700 ml-3 space-y-6 pb-2">
          
          <div className="relative pl-6">
            <div className="absolute w-3 h-3 bg-brand-500 rounded-full -left-[7px] top-1.5 shadow-[0_0_10px_rgba(20,184,166,0.6)]"></div>
            <h3 className="font-semibold text-brand-100">1. Audio File</h3>
            <p className="text-sm text-brand-300 mt-1">Raw .wav file uploaded by user</p>
          </div>

          <div className="relative pl-6">
            <div className="absolute w-3 h-3 bg-brand-600 rounded-full -left-[7px] top-1.5"></div>
            <h3 className="font-semibold text-brand-100">2. Audio Preprocessing</h3>
            <p className="text-sm text-brand-300 mt-1">Cropped/padded to 29s, amplitude normalized</p>
          </div>

          <div className="relative pl-6">
            <div className="absolute w-3 h-3 bg-brand-600 rounded-full -left-[7px] top-1.5"></div>
            <h3 className="font-semibold text-brand-100">3. Feature Extraction</h3>
            <p className="text-sm text-brand-300 mt-1">Converted to a 128-bin Log-Mel Spectrogram (2D image)</p>
          </div>

          <div className="relative pl-6">
            <div className="absolute w-3 h-3 bg-brand-600 rounded-full -left-[7px] top-1.5"></div>
            <h3 className="font-semibold text-brand-100">4. Trained ML/DL Model</h3>
            <p className="text-sm text-brand-300 mt-1">Passed through a 4-block PyTorch CNN</p>
          </div>

          <div className="relative pl-6">
            <div className="absolute w-3 h-3 bg-brand-500 rounded-full -left-[7px] top-1.5 shadow-[0_0_10px_rgba(20,184,166,0.6)]"></div>
            <h3 className="font-semibold text-brand-100">5. Genre Prediction</h3>
            <p className="text-sm text-brand-300 mt-1">Softmax output produces probability distribution</p>
          </div>
          
        </div>
      </div>

      {/* Model Information */}
      <div className="bg-brand-900/50 rounded-xl p-6 border border-brand-800 shadow-xl">
        <h2 className="text-xl font-bold mb-6 flex items-center gap-2 text-brand-100">
          <Database className="w-5 h-5 text-brand-500" />
          System Information
        </h2>
        
        <ul className="space-y-4">
          <li className="flex items-start gap-4 p-3 rounded-lg bg-brand-800/30">
            <Database className="w-5 h-5 text-brand-400 mt-0.5" />
            <div>
              <p className="font-semibold text-sm text-brand-200">Dataset</p>
              <p className="text-brand-50">GTZAN Dataset (1000 tracks)</p>
            </div>
          </li>
          
          <li className="flex items-start gap-4 p-3 rounded-lg bg-brand-800/30">
            <Music className="w-5 h-5 text-brand-400 mt-0.5" />
            <div>
              <p className="font-semibold text-sm text-brand-200">Classes</p>
              <p className="text-brand-50">10 genres (blues, classical, rock, etc.)</p>
            </div>
          </li>

          <li className="flex items-start gap-4 p-3 rounded-lg bg-brand-800/30">
            <Network className="w-5 h-5 text-brand-400 mt-0.5" />
            <div>
              <p className="font-semibold text-sm text-brand-200">Deep Learning Model</p>
              <p className="text-brand-50">Custom CNN (PyTorch) with SpecAugment</p>
            </div>
          </li>

          <li className="flex items-start gap-4 p-3 rounded-lg bg-brand-800/30">
            <Server className="w-5 h-5 text-brand-400 mt-0.5" />
            <div>
              <p className="font-semibold text-sm text-brand-200">Backend</p>
              <p className="text-brand-50">FastAPI with dynamic tensor preprocessing</p>
            </div>
          </li>

          <li className="flex items-start gap-4 p-3 rounded-lg bg-brand-800/30">
            <Layout className="w-5 h-5 text-brand-400 mt-0.5" />
            <div>
              <p className="font-semibold text-sm text-brand-200">Frontend</p>
              <p className="text-brand-50">React + Vite + TailwindCSS</p>
            </div>
          </li>
        </ul>
      </div>

    </div>
  );
}
