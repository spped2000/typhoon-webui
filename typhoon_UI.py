import os
import requests
import gradio as gr
from openai import OpenAI
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Load environment variables from .env file (for local development)
load_dotenv()

# Get API key from environment variable
TYPHOON_API_KEY = os.environ.get("TYPHOON_API_KEY")
if not TYPHOON_API_KEY:
    print("Warning: TYPHOON_API_KEY not found in environment variables.")
    print("Please set it in your .env file or in your Cloudflare environment variables.")

class URLSummarizer:
    """A class to summarize web content and output summaries in Thai with Markdown formatting."""
    
    def __init__(self):
        """Initialize the summarizer with API key from environment."""
        # Initialize the OpenAI client with the API key from environment
        self.client = OpenAI(
            api_key=TYPHOON_API_KEY,
            base_url='https://api.opentyphoon.ai/v1'
        )
        self.message_counter = 1
    
    def load_from_url(self, url):
        """Load content from a URL using BeautifulSoup for better text extraction."""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Parse the HTML with BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
                
            # Get text
            text = soup.get_text(separator=' ', strip=True)
            
            # Break into lines and remove leading/trailing space
            lines = (line.strip() for line in text.splitlines())
            # Break multi-headlines into a line each
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            # Remove blank lines
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            return text
        except Exception as e:
            return f"Error: {str(e)}"
    
    def chunk_text(self, text, chunk_size=10000):
        """Split text into manageable chunks."""
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            if current_size + len(word) + 1 > chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_size = len(word)
            else:
                current_chunk.append(word)
                current_size += len(word) + 1
                
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks
    
    def summarize_chunk(self, content, is_chunk=False, chunk_num=0, total_chunks=1):
        """Summarize a single chunk of content with Markdown formatting."""
        if not self.client:
            return "## ⚠️ ข้อผิดพลาด\nAPI key ไม่ถูกกำหนด"

        try:
            # Create message with appropriate context
            chunk_context = ""
            if is_chunk:
                chunk_context = f"This is part {chunk_num} of {total_chunks} of the full content. "
                
            messages = [
                {"role": "system", "content": "You are a professional summarizer. Create a comprehensive summary in Thai language using Markdown formatting for better readability."},
                {"role": "user", "content": f"""{chunk_context}Please summarize the following content according to these guidelines:

1. Focus on key points, main concepts, and important details
2. Use clear section headings with markdown (## for headings)
3. Use **bold** for emphasis on important terms
4. Use *italics* for names, titles, and secondary emphasis
5. Use bullet points or numbered lists where appropriate
6. Use markdown block quotes (>) for quotations or highlighted text
7. Format for easy reading with paragraph breaks
8. {'' if is_chunk else f"End with [End of Summary, Message #{self.message_counter}]"}

CONTENT TO SUMMARIZE:
{content}
"""}
            ]
            
            # Make API call
            response = self.client.chat.completions.create(
                model="typhoon-v2-70b-instruct", 
                messages=messages,
                max_tokens=2000,
                temperature=0.3
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"## ⚠️ ข้อผิดพลาด\nเกิดปัญหาในการสรุปส่วนที่ {chunk_num}: {str(e)}"
    
    def combine_summaries(self, summaries):
        """Combine multiple chunk summaries into one coherent summary with Markdown."""
        if not self.client:
            return "## ⚠️ ข้อผิดพลาด\nAPI key ไม่ถูกกำหนด"

        combined_text = "\n\n".join(summaries)
        
        try:
            messages = [
                {"role": "system", "content": "You are a professional summarizer. Create a comprehensive summary in Thai language using Markdown formatting for better readability."},
                {"role": "user", "content": f"""Please combine these summaries into one coherent summary according to these guidelines:

1. Ensure the summary flows naturally as one complete text
2. Remove any redundancies or repetitions
3. Ensure all key information is preserved
4. Use markdown formatting to enhance readability:
   - ## for section headings
   - **bold** for important terms
   - *italics* for names and titles
   - Bullet points or numbered lists where appropriate
   - > blockquotes for important quotes
5. Add a # main title at the top
6. End with [End of Summary, Message #{self.message_counter}]

SUMMARIES TO COMBINE:
{combined_text}
"""}
            ]
            
            # Make API call
            response = self.client.chat.completions.create(
                model="typhoon-v2-70b-instruct", 
                messages=messages,
                max_tokens=2000,
                temperature=0.3
            )
            
            self.message_counter += 1
            return response.choices[0].message.content
        except Exception as e:
            return f"## ⚠️ ข้อผิดพลาด\nเกิดปัญหาในการรวมบทสรุป: {str(e)}"
    
    def summarize_url(self, url, progress=gr.Progress()):
        """Process a URL and generate a summary with Markdown formatting."""
        # Check if API client is initialized
        if not TYPHOON_API_KEY:
            return "## ⚠️ ข้อผิดพลาด\nไม่พบ API key ในการตั้งค่าสภาพแวดล้อม กรุณาตั้งค่า TYPHOON_API_KEY"
        
        # Progress indicator
        progress(0, desc="กำลังโหลดข้อมูลจาก URL...")
        
        # Validate URL
        if not url:
            return "## ⚠️ ข้อผิดพลาด\nกรุณากรอก URL"
        
        # Add https:// if missing
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        # Load content from URL
        content = self.load_from_url(url)
        
        if not content or content.startswith("Error:"):
            return content if content else "## ⚠️ ข้อผิดพลาด\nไม่สามารถดึงข้อมูลจาก URL ได้"
        
        # Check if content is empty
        if not content.strip():
            return "## ⚠️ ข้อผิดพลาด\nไม่พบเนื้อหาจาก URL ที่ระบุ"
        
        # Split into chunks if needed
        chunks = self.chunk_text(content)
        progress(0.2, desc="โหลดข้อมูลสำเร็จ")
        
        summary = None
        
        if len(chunks) == 1:
            # Only one chunk, process directly
            progress(0.4, desc="กำลังสรุปเนื้อหา...")
            summary = self.summarize_chunk(chunks[0])
            progress(0.9, desc="สรุปเนื้อหาเสร็จสิ้น!")
        else:
            # Multiple chunks, process each and then combine
            progress(0.3, desc=f"เนื้อหายาวเกินไป แบ่งออกเป็น {len(chunks)} ส่วน")
            
            chunk_summaries = []
            for i, chunk in enumerate(chunks):
                progress_val = 0.3 + (0.5 * (i / len(chunks)))
                progress(progress_val, desc=f"กำลังสรุปส่วนที่ {i+1}/{len(chunks)}...")
                summary = self.summarize_chunk(chunk, True, i+1, len(chunks))
                if summary and not summary.startswith("## ⚠️ ข้อผิดพลาด"):
                    chunk_summaries.append(summary)
            
            if not chunk_summaries:
                return "## ⚠️ ข้อผิดพลาด\nไม่สามารถสรุปเนื้อหาได้"
                
            if len(chunk_summaries) == 1:
                summary = chunk_summaries[0]
            else:
                # Combine all summaries
                progress(0.8, desc="กำลังรวมบทสรุปทั้งหมด...")
                summary = self.combine_summaries(chunk_summaries)
            
            progress(1.0, desc="สรุปเนื้อหาเสร็จสิ้น!")
        
        if summary and not summary.startswith("## ⚠️ ข้อผิดพลาด"):
            return summary
        else:
            return summary if summary else "## ⚠️ ข้อผิดพลาด\nไม่สามารถสรุปเนื้อหาได้"


# Set up Gradio interface
def create_interface():
    summarizer = URLSummarizer()
    
    with gr.Blocks(title="Thai URL Summarizer", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🇹🇭 Typhoon2 สรุปเนื้อหาจากเว็บไซต์ by เพจ ตื่นมาโค้ดpython")
        gr.Markdown("ป้อน URL และรับบทสรุปที่ครอบคลุมเป็นภาษาไทย")
        
        with gr.Row():
            url_input = gr.Textbox(
                label="URL", 
                placeholder="ใส่ URL เว็บไซต์ที่นี่ (เช่น https://example.com)",
                lines=1
            )
        
        with gr.Row():
            submit_btn = gr.Button("สร้างบทสรุป", variant="primary")
            
        with gr.Row():
            output = gr.Markdown(
                label="บทสรุป (ภาษาไทย)",
                value="บทสรุปจะปรากฏที่นี่"
            )
        
        with gr.Accordion("เกี่ยวกับเครื่องมือนี้", open=False):
            gr.Markdown("""
            ## เกี่ยวกับเครื่องมือนี้
            
            เครื่องมือนี้ใช้โมเดล Typhoon v2-70b-instruct เพื่อสร้างบทสรุปที่ครอบคลุมของเนื้อหาเว็บในภาษาไทย พร้อมการจัดรูปแบบ Markdown เพื่อความอ่านง่าย
            
            ### คุณสมบัติ:
            - ประมวลผลเนื้อหาเว็บเพจใดก็ได้
            - รองรับบทความยาวโดยการแบ่งออกเป็นส่วนย่อยและรวมบทสรุปเข้าด้วยกัน
            - สร้างบทสรุปภาษาไทยที่มีโครงสร้างดีด้วยการจัดรูปแบบที่เหมาะสม
            - รักษาข้อมูลสำคัญจากเนื้อหาต้นฉบับ
            
            ### ข้อกำหนด:
            - ต้องมีการเชื่อมต่ออินเทอร์เน็ตเพื่อเข้าถึงเนื้อหา URL
            
            ### ประกาศความเป็นส่วนตัว:
            - เนื้อหา URL จะถูกประมวลผลชั่วคราวและไม่ถูกจัดเก็บอย่างถาวร
            """)
        
        gr.Markdown("""
        ### ตัวอย่างการใช้งาน
        ลองใส่ URL ของเว็บไซต์ที่มีเนื้อหาภาษาไทย เช่น บทความข่าว, บล็อก, หรือเพจสินค้า แล้วคลิก "สร้างบทสรุป"
        
        บทสรุปที่ได้จะรวมเนื้อหาสำคัญจากหน้าเว็บและนำเสนอในรูปแบบ Markdown พร้อมหัวข้อและการจัดรูปแบบที่อ่านง่าย
        """)
        
        # Set up the submission action
        submit_btn.click(
            fn=summarizer.summarize_url,
            inputs=[url_input],
            outputs=output
        )
    
    return interface

# Launch the Gradio app
if __name__ == "__main__":
    app = create_interface()
    app.launch(share=True)