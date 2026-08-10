import React, { useState } from 'react';
import { Headphones } from 'lucide-react';
import UploadArea from './components/UploadArea';
import ResultsDisplay from './components/ResultsDisplay';
import InfoSection from './components/InfoSection';

function App() {
  const [predictionResult, setPredictionResult] = useState(null);

  const handlePrediction = (data) => {
    setPredictionResult(data);
    // Scroll smoothly to results
    setTimeout(() => {
      window.scrollTo({ top: 300, behavior: 'smooth' });
    }, 100);
  };

  const handleReset = () => {
    setPredictionResult(null);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <div className="min-h-screen pb-20 selection:bg-brand-500/30">
      
      {/* Header */}
      <header className="pt-16 pb-12 px-4 text-center">
        <div className="inline-flex items-center justify-center p-3 bg-brand-800/50 rounded-2xl mb-6 shadow-xl border border-brand-700/50">
          <Headphones className="w-10 h-10 text-brand-400" />
        </div>
        <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight mb-4 text-transparent bg-clip-text bg-gradient-to-r from-white to-brand-200">
          GenreGuess
        </h1>
        <p className="text-lg md:text-xl text-brand-300 max-w-2xl mx-auto font-light">
          AI-powered music genre classification. Upload a track to analyze its acoustic features and predict its genre using a Deep Learning CNN.
        </p>
      </header>

      {/* Main Content Area */}
      <main className="px-4 relative z-10 flex flex-col items-center">
        
        {!predictionResult ? (
          <UploadArea onPrediction={handlePrediction} />
        ) : (
          <ResultsDisplay result={predictionResult} onReset={handleReset} />
        )}

        <InfoSection />

      </main>
      
      {/* Background decoration */}
      <div className="fixed inset-0 -z-10 overflow-hidden pointer-events-none">
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] rounded-full bg-brand-900/40 blur-[120px]"></div>
        <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] rounded-full bg-brand-800/20 blur-[120px]"></div>
      </div>
    </div>
  );
}

export default App;
