import os
import sys
import numpy as np
import warnings
import math
import textwrap
import glob

# Suppress warnings
warnings.filterwarnings("ignore")

# Check if whisper is installed, if not handle gracefully (though requirements.txt should handle it)
try:
    import whisper
except ImportError:
    print("CRITICAL: 'openai-whisper' not found. Please install it.")
    sys.exit(1)

from moviepy.editor import VideoClip, AudioFileClip
from PIL import Image, ImageDraw, ImageFont

class AutoCaptionAnimator:
    def __init__(self, image_path, audio_path, output_path, fps=30, sensitivity=0.20, min_threshold=0.01):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at: {image_path}")
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio not found at: {audio_path}")

        print(f"✅ Found assets: {image_path} | {audio_path}")

        self.image_path = image_path
        self.output_path = output_path
        self.fps = fps
        self.sensitivity = sensitivity
        self.min_threshold = min_threshold
        
        # --- STEP 1: LOAD AUDIO ---
        print("Loading audio...")
        try:
            self.audio_clip = AudioFileClip(audio_path)
            self.duration = self.audio_clip.duration
        except Exception as e:
            print(f"❌ Error loading audio: {e}")
            sys.exit(1)
        
        # --- STEP 2: AI TRANSCRIPTION (Whisper) ---
        print("🤖 AI is listening (Generating Subtitles)...")
        # Load 'base' model - fits in GitHub Actions memory
        try:
            model = whisper.load_model("base") 
            result = model.transcribe(audio_path, fp16=False)
            self.segments = result['segments']
            print(f"✅ Generated {len(self.segments)} subtitle segments.")
        except Exception as e:
            print(f"⚠️ Whisper error: {e}. Proceeding without subtitles.")
            self.segments = []

        # --- STEP 3: AUDIO ANALYSIS (Volume) ---
        print("Analyzing volume dynamics...")
        audio_fps = 44100
        self.audio_array = []
        try:
            for chunk in self.audio_clip.iter_chunks(fps=audio_fps, chunksize=audio_fps):
                if chunk.ndim > 1: chunk = chunk.mean(axis=1)
                self.audio_array.append(chunk)
            
            if self.audio_array:
                raw_audio = np.concatenate(self.audio_array)
            else:
                raw_audio = np.zeros(int(self.duration * audio_fps))
        except:
            raw_audio = np.zeros(int(self.duration * audio_fps))

        abs_audio = np.abs(raw_audio)
        
        # Downsample to Video FPS
        total_video_frames = int(self.duration * self.fps) + 1
        samples_per_frame = int(audio_fps / self.fps)
        
        self.volume_per_frame = []
        for i in range(total_video_frames):
            start = i * samples_per_frame
            end = start + samples_per_frame
            if start >= len(abs_audio):
                self.volume_per_frame.append(0)
                continue
            chunk = abs_audio[start:end]
            self.volume_per_frame.append(np.max(chunk) if len(chunk) > 0 else 0)
        
        self.volume_per_frame = np.array(self.volume_per_frame)
        
        # Normalize
        max_vol = np.max(self.volume_per_frame)
        if max_vol > 0: self.volume_per_frame /= max_vol
            
        # Smooth volume
        window_size = 3
        if len(self.volume_per_frame) >= window_size:
            self.volume_per_frame = np.convolve(self.volume_per_frame, np.ones(window_size)/window_size, mode='same')

        # --- STEP 4: PREPARE GRAPHICS ---
        self.original_pil_image = Image.open(image_path).convert("RGBA")
        self.base_width, self.base_height = self.original_pil_image.size
        
        max_scale = 1.0 + self.sensitivity
        self.canvas_w = int(self.base_width * max_scale * 1.4)
        self.canvas_h = int(self.base_height * max_scale * 1.4)
        
        # Font Loading Strategy
        self.font = None
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", # GitHub Actions Standard
            "arial.ttf",
            "Arial.ttf"
        ]
        
        for p in font_paths:
            try:
                self.font = ImageFont.truetype(p, 40)
                print(f"✅ Loaded font: {p}")
                break
            except:
                continue
        
        if self.font is None:
            print("⚠️ Warning: No custom font found. Using default.")
            self.font = ImageFont.load_default()

    def _get_subtitle_text(self, t):
        # Find current subtitle
        for seg in self.segments:
            if seg['start'] <= t <= seg['end']:
                return seg['text'].strip()
        return ""

    def _draw_text_with_stroke(self, draw, text, x, y, font, text_color, stroke_color, stroke_width):
        for off_x in range(-stroke_width, stroke_width + 1):
            for off_y in range(-stroke_width, stroke_width + 1):
                draw.text((x + off_x, y + off_y), text, font=font, fill=stroke_color)
        draw.text((x, y), text, font=font, fill=text_color)

    def _make_frame(self, t):
        # 1. Physics & Volume
        frame_idx = int(t * self.fps)
        vol = self.volume_per_frame[frame_idx] if frame_idx < len(self.volume_per_frame) else 0
        
        if vol > self.min_threshold:
            # Squash & Stretch
            stretch = 1.0 + (vol * self.sensitivity)
            squash = 1.0 - (vol * (self.sensitivity/2))
            rotation_val = math.sin(t * 15) * (vol * 4)
        else:
            # Idle breathing
            stretch = 1.0 + (math.sin(t * 2) * 0.01)
            squash = 1.0 + (math.sin(t * 2 + 1) * 0.01)
            rotation_val = 0

        # 2. Transform Sprite
        new_w = int(self.base_width * squash)
        new_h = int(self.base_height * stretch)
        resized = self.original_pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        if abs(rotation_val) > 0.1:
            resized = resized.rotate(rotation_val, resample=Image.Resampling.BICUBIC, expand=True)

        # 3. Dynamic Background
        bg_boost = int(vol * 40)
        bg_color = (20 + bg_boost, 20 + bg_boost, 30 + bg_boost)
        img = Image.new("RGB", (self.canvas_w, self.canvas_h), bg_color)
        
        # 4. Paste Sprite
        bg_w, bg_h = img.size
        sp_w, sp_h = resized.size
        anchor_y = int(self.canvas_h * 0.85) 
        pos_x = (bg_w - sp_w) // 2
        pos_y = anchor_y - sp_h
        img.paste(resized, (pos_x, pos_y), resized)

        # 5. Draw Subtitles
        text = self._get_subtitle_text(t)
        if text:
            draw = ImageDraw.Draw(img)
            
            # Text Wrapping
            char_width = 20 # Approx
            chars_per_line = self.canvas_w // char_width
            lines = textwrap.wrap(text, width=min(30, chars_per_line))
            
            line_height = 50
            text_y_start = self.canvas_h - 40 - (len(lines) * line_height)
            
            for i, line in enumerate(lines):
                bbox = draw.textbbox((0, 0), line, font=self.font)
                text_w = bbox[2] - bbox[0]
                text_x = (self.canvas_w - text_w) // 2
                text_y = text_y_start + (i * line_height)
                
                self._draw_text_with_stroke(draw, line, text_x, text_y, self.font, 
                                          text_color=(255, 230, 0), 
                                          stroke_color=(0, 0, 0), 
                                          stroke_width=3)
        
        return np.array(img)

    def render(self):
        print(f"Rendering {self.output_path} ({self.duration:.1f}s)...")
        anim_clip = VideoClip(self._make_frame, duration=self.duration)
        anim_clip = anim_clip.set_audio(self.audio_clip)
        
        anim_clip.write_videofile(
            self.output_path, 
            fps=self.fps, 
            codec="libx264", 
            audio_codec="aac",
            threads=2, # Safer for GH Actions
            preset="ultrafast" # Speed over size
        )
        print("Done. Maximum Impact achieved.")

if __name__ == "__main__":
    # --- AUTO-DETECTION LOGIC ---
    print(f"Current Working Directory: {os.getcwd()}")
    print("Files in Directory:", os.listdir('.'))

    # Try to find assets automatically if not passed
    img_file = None
    audio_file = None
    output_file = "final_video.mp4"

    # Find Audio
    audio_extensions = ['*.mp3', '*.wav', '*.m4a']
    for ext in audio_extensions:
        matches = glob.glob(ext)
        if matches:
            audio_file = matches[0]
            break

    # Find Image
    img_extensions = ['*.png', '*.jpg', '*.jpeg']
    for ext in img_extensions:
        matches = glob.glob(ext)
        if matches:
            img_file = matches[0]
            break

    # Override with arguments if provided
    if len(sys.argv) > 1 and sys.argv[1] != "dummy_arg": img_file = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] != "dummy_arg": audio_file = sys.argv[2]
    if len(sys.argv) > 3: output_file = sys.argv[3]

    if img_file and audio_file:
        try:
            animator = AutoCaptionAnimator(img_file, audio_file, output_file)
            animator.render()
        except Exception as e:
            print(f"CRITICAL ERROR: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        print("❌ Could not auto-detect input files.")
        print(f"Found Audio: {audio_file}")
        print(f"Found Image: {img_file}")
        sys.exit(1)
