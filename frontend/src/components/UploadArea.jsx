import React, { useState, useRef } from 'react';
import { UploadCloud, Music, FileAudio, Loader2 } from 'lucide-react';

export default function UploadArea({ onPrediction }) {
  const [file, setFile] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const audioRef = useRef(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const droppedFile = e.dataTransfer.files[0];
      validateAndSetFile(droppedFile);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      validateAndSetFile(e.target.files[0]);
    }
  };

  const validateAndSetFile = (selectedFile) => {
    setError(null);
    if (!selectedFile.name.toLowerCase().endsWith('.wav')) {
      setError("Please select a .wav file. Other formats are not supported.");
      setFile(null);
      return;
    }
    setFile(selectedFile);
  };

  const handlePredict = async () => {
    if (!file) return;

    setIsLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      // Use environment variable for backend URL
      const apiUrl = import.meta.env.VITE_API_URL;
      const response = await fetch(`${apiUrl}/predict`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Prediction failed');
      }

      const data = await response.json();
      onPrediction(data);
    } catch (err) {
      setError(err.message || "An error occurred while communicating with the server.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="w-full max-w-2xl mx-auto">
      <div
        className={`relative border-2 border-dashed rounded-2xl p-10 transition-all duration-300 ease-in-out text-center cursor-pointer ${isDragging
            ? 'border-brand-400 bg-brand-900/60 shadow-[0_0_20px_rgba(20,184,166,0.3)]'
            : 'border-brand-700 bg-brand-900/30 hover:bg-brand-900/50 hover:border-brand-500'
          }`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => !file && document.getElementById('file-upload').click()}
      >
        <input
          type="file"
          id="file-upload"
          className="hidden"
          accept=".wav,audio/wav"
          onChange={handleFileChange}
        />

        {!file ? (
          <div className="flex flex-col items-center justify-center space-y-4">
            <div className="w-20 h-20 rounded-full bg-brand-800/50 flex items-center justify-center mb-2 shadow-lg">
              <UploadCloud className="w-10 h-10 text-brand-400" />
            </div>
            <h3 className="text-xl font-semibold text-brand-100">Drag & drop your audio file</h3>
            <p className="text-brand-300 text-sm">or click to browse your computer</p>
            <div className="inline-flex items-center justify-center px-4 py-1.5 rounded-full bg-brand-900 border border-brand-700 text-xs text-brand-400 font-medium tracking-wide uppercase">
              Supports .WAV only
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center space-y-6" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center gap-4 bg-brand-800/60 px-6 py-4 rounded-xl border border-brand-700 w-full max-w-md">
              <FileAudio className="w-8 h-8 text-brand-400 shrink-0" />
              <div className="text-left overflow-hidden">
                <p className="text-brand-50 font-medium truncate" title={file.name}>{file.name}</p>
                <p className="text-brand-400 text-xs">{(file.size / (1024 * 1024)).toFixed(2)} MB</p>
              </div>
              <button
                onClick={() => setFile(null)}
                className="ml-auto text-brand-400 hover:text-red-400 transition-colors text-sm font-medium"
              >
                Change
              </button>
            </div>

            <div className="w-full max-w-md">
              <audio
                ref={audioRef}
                controls
                className="w-full h-10"
                src={URL.createObjectURL(file)}
              />
            </div>

            <button
              onClick={handlePredict}
              disabled={isLoading}
              className={`relative overflow-hidden group flex items-center justify-center gap-2 w-full max-w-md py-4 px-8 rounded-xl font-bold text-lg transition-all duration-300 ${isLoading
                  ? 'bg-brand-700 text-brand-200 cursor-not-allowed'
                  : 'bg-brand-500 hover:bg-brand-400 text-brand-950 shadow-[0_0_20px_rgba(20,184,166,0.4)] hover:shadow-[0_0_30px_rgba(20,184,166,0.6)] hover:-translate-y-1'
                }`}
            >
              {isLoading ? (
                <>
                  <Loader2 className="w-6 h-6 animate-spin" />
                  Analyzing Audio...
                </>
              ) : (
                <>
                  <Music className="w-6 h-6" />
                  Predict Genre
                </>
              )}
            </button>
          </div>
        )}
      </div>

      {error && (
        <div className="mt-4 p-4 rounded-lg bg-red-900/30 border border-red-800 text-red-200 text-sm text-center animate-pulse">
          {error}
        </div>
      )}
    </div>
  );
}
