import os
import sys
import numpy as np
import warnings
import math
import textwrap

# Suppress warnings
warnings.filterwarnings("ignore")

import whisper
from moviepy.editor import VideoClip, AudioFileClip
from PIL import Image, ImageDraw, ImageFont

class AutoCaptionAnimator:
    def __init__(self, image_path, audio_path, output_path, fps=30, sensitivity=0.20, min_threshold=0.01):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at: {image_path}")
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio not found at: {audio_path}")

        self.image_path = image_path
        self.output_path = output_path
        self.fps = fps
        self.sensitivity = sensitivity
        self.min_threshold = min_threshold
        
        # --- STEP 1: LOAD AUDIO ---
        print("Loading audio...")
        self.audio_clip = AudioFileClip(audio_path)
        self.duration = self.audio_clip.duration
        
        # --- STEP 2: AI TRANSCRIPTION (Whisper) ---
        print("🤖 AI is listening to audio (Generating Subtitles)...")
        # Load 'base' model - good balance of speed vs accuracy for CPU runners
        # Use 'small' or 'medium' if Shona accuracy is poor, but it takes longer.
        model = whisper.load_model("base") 
        
        # Transcribe
        result = model.transcribe(audio_path, fp16=False) # fp16=False for CPU
        self.segments = result['segments']
        
        print(f"✅ Generated {len(self.segments)} subtitle segments.")

        # --- STEP 3: AUDIO ANALYSIS (Volume) ---
        print("Analyzing volume dynamics...")
        audio_fps = 44100
        self.audio_array = []
        try:
            for chunk in self.audio_clip.iter_chunks(fps=audio_fps, chunksize=audio_fps):
                if chunk.ndim > 1: chunk = chunk.mean(axis=1)
                self.audio_array.append(chunk)
            raw_audio = np.concatenate(self.audio_array) if self.audio_array else np.zeros(int(self.duration * audio_fps))
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
        
        # Try to load a font, fallback to default if missing
        try:
            # Linux/GitHub Actions usually has DejaVuSans-Bold
            self.font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
        except:
            try:
                self.font = ImageFont.truetype("arial.ttf", 40)
            except:
                self.font = ImageFont.load_default()

    def _get_subtitle_text(self, t):
        # Linear search is fast enough for short videos
        for seg in self.segments:
            if seg['start'] <= t <= seg['end']:
                return seg['text'].strip()
        return ""

    def _draw_text_with_stroke(self, draw, text, x, y, font, text_color, stroke_color, stroke_width):
        # Manually draw stroke by drawing text at offsets
        for off_x in range(-stroke_width, stroke_width + 1):
            for off_y in range(-stroke_width, stroke_width + 1):
                draw.text((x + off_x, y + off_y), text, font=font, fill=stroke_color)
        draw.text((x, y), text, font=font, fill=text_color)

    def _make_frame(self, t):
        # 1. Physics & Volume
        frame_idx = int(t * self.fps)
        vol = self.volume_per_frame[frame_idx] if frame_idx < len(self.volume_per_frame) else 0
        
        if vol > self.min_threshold:
            stretch = 1.0 + (vol * self.sensitivity)
            squash = 1.0 - (vol * (self.sensitivity/2))
            rotation_val = math.sin(t * 15) * (vol * 4)
        else:
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
        anchor_y = int(self.canvas_h * 0.85) # Leave room for subtitles at bottom
        pos_x = (bg_w - sp_w) // 2
        pos_y = anchor_y - sp_h
        img.paste(resized, (pos_x, pos_y), resized)

        # 5. Draw Subtitles
        text = self._get_subtitle_text(t)
        if text:
            draw = ImageDraw.Draw(img)
            
            # Text Wrapping
            # Estimate char width approx to wrap
            char_width = 20 # rough estimate
            chars_per_line = self.canvas_w // char_width
            lines = textwrap.wrap(text, width=min(30, chars_per_line)) # Max 30 chars wide
            
            # Start drawing from bottom up
            line_height = 50
            text_y_start = self.canvas_h - 40 - (len(lines) * line_height)
            
            for i, line in enumerate(lines):
                # Calculate text size using bbox (robust)
                bbox = draw.textbbox((0, 0), line, font=self.font)
                text_w = bbox[2] - bbox[0]
                text_x = (self.canvas_w - text_w) // 2
                text_y = text_y_start + (i * line_height)
                
                # Draw Yellow Text with Black Stroke
                self._draw_text_with_stroke(draw, line, text_x, text_y, self.font, 
                                          text_color=(255, 230, 0), # Cyber Yellow
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
            threads=4,
            preset="fast" # Faster encode for CI
        )
        print("Done. Maximum Impact achieved.")

if __name__ == "__main__":
    # Standard Argument Parsing
    img_file = sys.argv[1] if len(sys.argv) > 1 else "character.png"
    audio_file = sys.argv[2] if len(sys.argv) > 2 else "reply.mp3"
    output_file = sys.argv[3] if len(sys.argv) > 3 else "output.mp4"
    
    # Auto-Search fallback (useful for local test)
    if not os.path.exists(img_file):
        for f in os.listdir('.'):
            if f.endswith(('.png', '.jpg')):
                img_file = f
                break
    if not os.path.exists(audio_file):
        for f in os.listdir('.'):
            if f.endswith('.mp3'):
                audio_file = f
                break
                
    try:
        animator = AutoCaptionAnimator(img_file, audio_file, output_file)
        animator.render()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
