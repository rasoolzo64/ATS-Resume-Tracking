from dotenv import load_dotenv
load_dotenv()
import os
import warnings

# Suppress the specific ALTS warnings
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'


import base64
import streamlit as st
import os
import io
import fitz  # PyMuPDF
import google.generativeai as genai
import json
import pandas as pd
from datetime import datetime
import time

# Load API key from Streamlit secrets or .env
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))

# Configure Gemini with API key
genai.configure(api_key=GOOGLE_API_KEY)

# Force API version to v1 globally
genai._default_version = "v1"

# === Enhanced Function Definitions ===

@st.cache_data(show_spinner=False)
def get_gemini_response(input_text, pdf_content, prompt, max_retries=3):
    """Enhanced Gemini response with error handling and retries"""
    for attempt in range(max_retries):
        try:
            model = genai.GenerativeModel("models/gemini-flash-latest")
            response = model.generate_content([
                {"text": input_text},
                pdf_content[0],
                {"text": prompt}
            ])
            return response.text
        except Exception as e:
            if attempt == max_retries - 1:
                st.error(f"❌ API Error: {str(e)}")
                return None
            time.sleep(2)  # Wait before retry

@st.cache_data(show_spinner=False)
def input_pdf_setup(uploaded_file):
    """Enhanced PDF processing with multiple pages support"""
    if uploaded_file is not None:
        try:
            pdf_bytes = uploaded_file.read()
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            
            # Process first 3 pages maximum to avoid token limits
            pdf_parts = []
            max_pages = min(3, len(doc))
            
            for page_num in range(max_pages):
                page = doc.load_page(page_num)
                pix = page.get_pixmap()
                
                img_byte_arr = io.BytesIO(pix.tobytes("jpeg"))
                img_bytes = img_byte_arr.getvalue()
                
                pdf_parts.append({
                    "mime_type": "image/jpeg",
                    "data": base64.b64encode(img_bytes).decode()
                })
            
            doc.close()
            return pdf_parts
        except Exception as e:
            st.error(f"❌ Error processing PDF: {str(e)}")
            return None
    else:
        st.error("❌ No file uploaded")
        return None

def extract_structured_data(response_text):
    """Extract structured data from AI response"""
    try:
        # Try to parse JSON if present
        if "{" in response_text and "}" in response_text:
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            json_str = response_text[start:end]
            return json.loads(json_str)
    except:
        pass
    return None

def calculate_score_visual(score):
    """Create visual score representation"""
    if score is None:
        return "N/A"
    
    color = "🟢" if score >= 80 else "🟡" if score >= 60 else "🔴"
    progress = "█" * int(score / 10) + "░" * (10 - int(score / 10))
    return f"{color} {score}% {progress}"

# === Session State Initialization ===
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'current_response' not in st.session_state:
    st.session_state.current_response = None
if 'current_analysis_type' not in st.session_state:
    st.session_state.current_analysis_type = None

# === Streamlit App UI Enhancement ===
st.set_page_config(
    page_title="🚀 ATS Resume Expert Pro",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem !important;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 800;
    }
    .main-header span.gradient-text {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: inline-block;
    }
    .main-header span.emoji {
        display: inline-block;
        margin-right: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .analysis-card {
        background: white;
        border-radius: 15px;
        padding: 1rem;
        border: 2px solid #e0e0e0;
        transition: all 0.3s ease;
        height: 50%; /* Make cards take full height */
        
        display: flex;
        flex-direction: column;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .analysis-card:hover {
        border-color: #667eea;
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    .analysis-card.selected {
        border-color: #667eea;
        background: linear-gradient(135deg, #f8f9ff 0%, #f0f2ff 100%);
    }
    
    /* Card content sections with flex grow */
    .card-header {
        flex-shrink: 0;
        text-align: center;
        margin-bottom: 1rem;
    }
    .card-features {
        flex-grow: 1;
        margin: 1rem 0;
    }
    .card-button {
        flex-shrink: 0;
        margin-top: auto;
    }
    
    .feature-list {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stButton button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    .stButton button:hover {
        transform: translateY(-2px) !important;
    }
    .success-box {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
    }
    .info-box {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #17a2b8;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <h2 style='color: #667eea;'>⚙️ ATS Pro</h2>
        <p style='color: #666; font-size: 0.9rem;'>AI-Powered Resume Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("🔧 Configuration")
    
    st.markdown("### 📊 Analysis Settings")
    save_results = st.toggle("💾 Save to History", value=True, help="Store analysis results in history")
    enable_debug = st.toggle("🐛 Debug Mode", value=False, help="Show detailed processing information")
    
    st.markdown("---")
    
    st.markdown("### 📈 Features")
    col_feat1, col_feat2 = st.columns(2)
    with col_feat1:
        st.checkbox("🔍 Keyword Analysis", value=True)
        st.checkbox("📝 Format Check", value=True)
    with col_feat2:
        st.checkbox("🎯 ATS Score", value=True)
        st.checkbox("💡 Suggestions", value=True)
    
    
    st.markdown("---")
    
    st.markdown("### ℹ️ About")
    with st.expander("How it works"):
        st.info("""
        **ATS Resume Expert Pro** uses Google Gemini AI to:
        
        • 📊 Analyze resume-job match
        • 🔍 Identify missing keywords  
        • 💡 Provide improvement tips
        • 🎯 Calculate ATS compatibility
        • 📈 Generate detailed reports
        """)

# Main Header
st.markdown('<h1 class="main-header">🚀 ATS Resume Expert Pro</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #666; font-size: 1.2rem; margin-bottom: 3rem;">AI-Powered Resume Analysis & ATS Optimization</p>', unsafe_allow_html=True)

# Main Layout
col1, col2 = st.columns([1, 1])

with col1:
    # Job Description Section
    st.markdown("### 📋 Job Description")
    with st.container():
        input_text = st.text_area(
            "**Paste the job description here:**",
            height=200,
            placeholder="Copy and paste the complete job description...\n\nExample:\n• Required skills: Python, SQL, Machine Learning\n• Experience: 3+ years in data analysis\n• Education: Bachelor's in Computer Science\n• Responsibilities: Data modeling, reporting, insights generation",
            key="input",
            label_visibility="collapsed"
        )
    
    # Resume Upload Section
    st.markdown("### 📄 Resume Upload")
    with st.container():
        uploaded_file = st.file_uploader(
            "**Upload your resume (PDF):**",
            type=["pdf"],
            help="Supported: PDF files up to 10MB",
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            # File details card
            st.markdown("""
            <div style='
                background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
                padding: 1rem;
                border-radius: 10px;
                border-left: 4px solid #28a745;
                margin: 1rem 0;
            '>
            """, unsafe_allow_html=True)
            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.metric("📄 File", uploaded_file.name)
                st.metric("📊 Size", f"{uploaded_file.size / 1024:.1f} KB")
            with col_info2:
                st.metric("✅ Status", "Uploaded")
                st.metric("🕒 Time", datetime.now().strftime("%H:%M"))
            st.markdown("</div>", unsafe_allow_html=True)

with col2:
    # Analysis Type Selection - Card Layout
    st.markdown("### 🎯 Choose Analysis Type")
    st.markdown("<p style='color: #666; margin-bottom: 1.5rem;'>Select the depth of analysis based on your needs</p>", unsafe_allow_html=True)
    
    # Analysis Options Configuration
    analysis_options = [
        {
            "icon": "🚀",
            "title": "Quick Scan", 
            "description": "Fast 30-second overview",
            "color": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
            "features": ["Basic match assessment", "Key strengths/weaknesses", "Quick recommendations", "Instant results"],
            "prompt": """
            QUICK SCAN - Provide concise evaluation (under 250 words):
            
            🎯 **Overall Match**: [Good/Fair/Poor fit]
            
            ✅ **Top 3 Strengths**:
            - [Most relevant qualification]
            - [Key experience match]
            - [Main skill alignment]
            
            ⚠️ **Top 3 Weaknesses**:
            - [Major gap]
            - [Missing requirement]
            - [Area for improvement]
            
            💡 **Quick Recommendations**: [2-3 actionable tips]
            
            Be direct, concise, and actionable.
            """,
            "button_type": "secondary"
        },
        {
            "icon": "📊", 
            "title": "Detailed Analysis",
            "description": "Comprehensive ATS evaluation", 
            "color": "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)",
            "features": ["ATS compatibility score", "Match percentage", "Missing keywords", "Detailed insights", "Structured report"],
            "prompt": """
            DETAILED ANALYSIS - Provide structured evaluation with metrics:
            
            Please respond in this EXACT JSON format:
            {
                "match_percentage": 85,
                "ats_score": 78,
                "overall_assessment": "Brief summary of fit",
                "missing_keywords": ["Python", "Machine Learning", "AWS", "SQL", "Agile"],
                "strengths": ["Strong educational background", "Relevant project experience", "Technical skills match"],
                "weaknesses": ["Missing certification", "Limited leadership experience", "Gap in specific technology"],
                "recommendations": ["Add missing keywords strategically", "Quantify achievements", "Highlight relevant projects"]
            }
            
            If JSON is not possible, use clear headings and structure.
            """,
            "button_type": "primary"
        },
        {
            "icon": "💎",
            "title": "Improvement Pro", 
            "description": "Expert optimization guide",
            "color": "linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)",
            "features": ["ATS optimization", "Content enhancement", "Professional tips", "Best practices", "Competitive edge"],
            "prompt": """
            IMPROVEMENT PRO - Provide expert resume optimization:
            
            🎯 **Executive Summary**: Overall assessment and potential
            
            🔍 **ATS Optimization**:
            • Keyword analysis and placement
            • Formatting improvements for ATS
            • Section optimization tips
            
            💡 **Content Enhancement**:
            • Achievement quantification
            • Action verb suggestions
            • Impact statement improvements
            
            📊 **Professional Recommendations**:
            • Skill highlighting strategy
            • Experience reframing
            • Competitive positioning
            
            🚀 **Quick Wins**: Immediate improvements (3-5 items)
            📈 **Long-term Strategy**: Career development suggestions
            
            Provide specific, actionable advice.
            """,
            "button_type": "secondary"
        }
    ]

    # Create analysis cards
    cols = st.columns(3)
    
    for i, option in enumerate(analysis_options):
        with cols[i]:
            # Determine if this card is selected
            is_selected = st.session_state.get('current_analysis_type') == option['title']
            
            # Card container with enhanced styling
            st.markdown(f"""
            <div class="analysis-card {'selected' if is_selected else ''}" 
                 style="border-color: {'#667eea' if is_selected else '#e0e0e0'};">
                <div style="text-align: center; margin-bottom: 1rem;">
                    <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">{option['icon']}</div>
                    <h3 style="color: #2c3e50; margin: 0.5rem 0; font-weight: 700;">{option['title']}</h3>
                    <p style="color: #666; font-size: 0.9rem; margin: 0;">{option['description']}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Features list
            with st.expander("📋 What's included", expanded=False):
                for feature in option['features']:
                    st.markdown(f"✓ {feature}")
            
            # Analysis button
            if st.button(
                f"Run {option['title']}",
                use_container_width=True,
                type=option['button_type'],
                key=f"btn_{option['title'].replace(' ', '_').lower()}"
            ):
                st.session_state.current_analysis_type = option['title']
                st.session_state.selected_prompt = option['prompt']
                st.rerun()

# Analysis Execution Section
if uploaded_file is not None and st.session_state.get('current_analysis_type'):
    st.markdown("---")
    
    # Show current analysis info
    current_type = st.session_state.current_analysis_type
    st.markdown(f"### 🔍 Running: {current_type}")
    
    with st.spinner(f"**Analyzing your resume with {current_type}...** This may take 15-30 seconds."):
        pdf_content = input_pdf_setup(uploaded_file)
        
        if pdf_content:
            response = get_gemini_response(
                input_text, 
                pdf_content, 
                st.session_state.selected_prompt
            )
            
            if response:
                st.session_state.current_response = response
                
                # Save to history
                if save_results:
                    history_entry = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "analysis_type": current_type,
                        "response": response[:500] + "..." if len(response) > 500 else response
                    }
                    st.session_state.analysis_history.append(history_entry)
                
                # Display Results
                st.markdown("---")
                st.markdown(f"### 📋 {current_type} Results")
                
                # Enhanced Results Display
                if current_type == "Detailed Analysis":
                    structured_data = extract_structured_data(response)
                    
                    if structured_data and 'match_percentage' in structured_data:
                        # Metrics Dashboard
                        st.markdown("#### 📊 Metrics Dashboard")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4 style='color: white; margin: 0;'>🎯 Overall Match</h4>
                                <h2 style='color: white; margin: 0.5rem 0;'>{calculate_score_visual(structured_data['match_percentage'])}</h2>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4 style='color: white; margin: 0;'>🤖 ATS Score</h4>
                                <h2 style='color: white; margin: 0.5rem 0;'>{calculate_score_visual(structured_data.get('ats_score'))}</h2>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col3:
                            missing_count = len(structured_data.get('missing_keywords', []))
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4 style='color: white; margin: 0;'>⚠️ Missing Keywords</h4>
                                <h2 style='color: white; margin: 0.5rem 0;'>{missing_count}</h2>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col4:
                            strengths_count = len(structured_data.get('strengths', []))
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4 style='color: white; margin: 0;'>✅ Strengths</h4>
                                <h2 style='color: white; margin: 0.5rem 0;'>{strengths_count}</h2>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Detailed Sections
                        if 'missing_keywords' in structured_data and structured_data['missing_keywords']:
                            with st.expander("🔍 Missing Keywords Analysis", expanded=True):
                                st.write("**Keywords to add to your resume:**")
                                keywords_html = " ".join([f"<span style='background: #ff6b6b; color: white; padding: 0.3rem 0.6rem; border-radius: 20px; margin: 0.2rem; display: inline-block;'>{kw}</span>" for kw in structured_data['missing_keywords'][:10]])
                                st.markdown(keywords_html, unsafe_allow_html=True)
                        
                        if 'strengths' in structured_data and structured_data['strengths']:
                            with st.expander("✅ Key Strengths", expanded=True):
                                for strength in structured_data['strengths']:
                                    st.markdown(f"🎯 {strength}")
                        
                        if 'recommendations' in structured_data and structured_data['recommendations']:
                            with st.expander("💡 Actionable Recommendations", expanded=True):
                                for i, rec in enumerate(structured_data['recommendations'], 1):
                                    st.markdown(f"{i}. **{rec}**")
                
                # Full Response Display
                with st.expander("📝 Detailed Analysis Report", expanded=True):
                    st.markdown("#### Complete Analysis")
                    st.write(response)
                
                # Download Section
                st.markdown("---")
                col_dl1, col_dl2 = st.columns([3, 1])
                with col_dl1:
                    st.markdown("#### 💾 Save Results")
                with col_dl2:
                    st.download_button(
                        label="📥 Download Full Report",
                        data=response,
                        file_name=f"resume_analysis_{current_type.replace(' ', '_').lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )

# Analysis History
if st.session_state.analysis_history:
    st.markdown("---")
    st.markdown("### 📚 Analysis History")
    st.markdown("<p style='color: #666;'>Your recent resume analyses</p>", unsafe_allow_html=True)
    
    for i, entry in enumerate(reversed(st.session_state.analysis_history[-5:]), 1):
        with st.expander(f"**Analysis {i}** - {entry['timestamp']} ({entry['analysis_type']})", expanded=False):
            st.markdown(f"**Type:** {entry['analysis_type']}")
            st.markdown(f"**Time:** {entry['timestamp']}")
            st.markdown("**Summary:**")
            st.write(entry['response'])

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='
        text-align: center; 
        color: #666; 
        padding: 2rem 0;
        background: linear-gradient(135deg, #f8f9ff 0%, #f0f2ff 100%);
        border-radius: 15px;
        margin-top: 2rem;
    '>
        <h4 style='color: #667eea; margin-bottom: 1rem;'>🚀 ATS Resume Expert Pro</h4>
        <p style='margin: 0.5rem 0;'>Built with ❤️ using Google Gemini AI & Streamlit</p>
        <p style='margin: 0.5rem 0; font-size: 0.9rem; color: #888;'>
        Note: This AI-powered tool provides suggestions. Always verify important results manually.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Clear analysis button
if st.session_state.get('current_analysis_type'):
    if st.button("🔄 Clear Current Analysis", use_container_width=True):
        st.session_state.current_analysis_type = None
        st.session_state.current_response = None
        st.rerun()