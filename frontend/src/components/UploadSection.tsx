import { Upload, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { useToast } from "@/hooks/use-toast";

interface UploadSectionProps {
  videoFile: File | null;
  setVideoFile: (file: File | null) => void;
  textDescription: string;
  setTextDescription: (text: string) => void;
  onAnalyze: () => void;
  isAnalyzing: boolean;
}

export const UploadSection = ({
  videoFile,
  setVideoFile,
  textDescription,
  setTextDescription,
  onAnalyze,
  isAnalyzing
}: UploadSectionProps) => {
  const { toast } = useToast();

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      if (file.size > 100 * 1024 * 1024) {
        toast({
          title: "File too large",
          description: "Please upload a video under 100MB",
          variant: "destructive"
        });
        return;
      }
      setVideoFile(file);
    }
  };

  const handleAnalyzeClick = () => {
    if (!videoFile) {
      toast({
        title: "Video required",
        description: "Please upload a video file",
        variant: "destructive"
      });
      return;
    }
    if (!textDescription.trim()) {
      toast({
        title: "Description required",
        description: "Please provide a text description",
        variant: "destructive"
      });
      return;
    }
    onAnalyze();
  };

  return (
    <div className="mx-auto max-w-4xl space-y-6">
      <Card className="shadow-lg">
        <CardHeader>
          <CardTitle className="text-2xl">Upload Evidence</CardTitle>
          <CardDescription>
            Upload your video clip (10-30 seconds) and provide a text description of what should be in the video.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="space-y-2">
            <Label htmlFor="video-upload" className="text-base font-semibold">
              Video File
            </Label>
            <div className="flex flex-col gap-3">
              <div className="relative">
                <input
                  id="video-upload"
                  type="file"
                  accept="video/*"
                  onChange={handleFileChange}
                  className="hidden"
                />
                <label
                  htmlFor="video-upload"
                  className="flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed border-border bg-muted/50 p-12 transition-colors hover:border-accent hover:bg-accent/5"
                >
                  <Upload className="mb-4 h-12 w-12 text-muted-foreground" />
                  {videoFile ? (
                    <div className="text-center">
                      <p className="font-medium text-foreground">{videoFile.name}</p>
                      <p className="text-sm text-muted-foreground">
                        {(videoFile.size / (1024 * 1024)).toFixed(2)} MB
                      </p>
                    </div>
                  ) : (
                    <div className="text-center">
                      <p className="font-medium text-foreground">Click to upload video</p>
                      <p className="text-sm text-muted-foreground">MP4, MOV, AVI up to 100MB</p>
                    </div>
                  )}
                </label>
              </div>
            </div>
          </div>

          <div className="space-y-2">
            <Label htmlFor="text-description" className="text-base font-semibold">
              Text Description
            </Label>
            <Textarea
              id="text-description"
              placeholder="Enter claims to verify (e.g., 'Three people, two cars, no weapons visible')"
              value={textDescription}
              onChange={(e) => setTextDescription(e.target.value)}
              className="min-h-32 resize-none"
            />
            <p className="text-xs text-muted-foreground">
              Supported claims: Number of people, vehicles, and weapon presence/absence
            </p>
          </div>

          <Button
            onClick={handleAnalyzeClick}
            disabled={isAnalyzing || !videoFile || !textDescription}
            size="lg"
            className="w-full bg-gradient-accent font-semibold shadow-glow transition-all hover:scale-[1.02]"
          >
            {isAnalyzing ? (
              <>
                <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                Analyzing Evidence...
              </>
            ) : (
              "Analyze Evidence"
            )}
          </Button>
        </CardContent>
      </Card>
    </div>
  );
};
