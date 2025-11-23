import { useState } from "react";
import { Hero } from "@/components/Hero";
import { UploadSection } from "@/components/UploadSection";
import { ResultsSection } from "@/components/ResultsSection";

export interface AnalysisResult {
  overallScore: number;
  claims: {
    type: "people" | "vehicles" | "weapons";
    claimed: string;
    detected: string;
    consistent: boolean;
    confidence: number;
  }[];
}

const Index = () => {
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [textDescription, setTextDescription] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState<AnalysisResult | null>(null);

  const handleAnalyze = async () => {
    if (!videoFile || !textDescription) {
      return;
    }

    setIsAnalyzing(true);
    
    // Simulate API call - In production, this would call the backend
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // Mock results for demo
    const mockResults: AnalysisResult = {
      overallScore: 85,
      claims: [
        {
          type: "people",
          claimed: "3 people",
          detected: "3 people",
          consistent: true,
          confidence: 95
        },
        {
          type: "vehicles",
          claimed: "2 cars",
          detected: "2 cars",
          consistent: true,
          confidence: 92
        },
        {
          type: "weapons",
          claimed: "No weapons",
          detected: "No weapons",
          consistent: true,
          confidence: 88
        }
      ]
    };
    
    setResults(mockResults);
    setIsAnalyzing(false);
  };

  return (
    <div className="min-h-screen bg-background">
      <Hero />
      <main className="container mx-auto px-4 py-12">
        <UploadSection
          videoFile={videoFile}
          setVideoFile={setVideoFile}
          textDescription={textDescription}
          setTextDescription={setTextDescription}
          onAnalyze={handleAnalyze}
          isAnalyzing={isAnalyzing}
        />
        {results && <ResultsSection results={results} />}
      </main>
    </div>
  );
};

export default Index;
