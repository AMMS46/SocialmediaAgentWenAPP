import streamlit as st
import pandas as pd
import os
import requests
import re
from datetime import datetime, timedelta
from dotenv import load_dotenv
import json
import time
import logging
from typing import Optional, Dict, List, Tuple
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Social Media Content Agent",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def hash_string(text: str) -> str:
    """Create a hash of text for caching"""
    return hashlib.md5(text.encode()).hexdigest()

def validate_api_keys() -> Dict[str, bool]:
    """Validate all API keys and return status"""
    return {
        "gemini": bool(os.getenv("GOOGLE_API_KEY")),
        "buffer": bool(os.getenv("BUFFER_ACCESS_TOKEN")),
        "unsplash": bool(os.getenv("UNSPLASH_ACCESS_KEY"))
    }

def safe_get_session_state(key: str, default=None):
    """Safely get session state with error handling"""
    try:
        return st.session_state.get(key, default)
    except Exception as e:
        logger.error(f"Error accessing session state for key {key}: {e}")
        return default

def safe_set_session_state(key: str, value):
    """Safely set session state with error handling"""
    try:
        st.session_state[key] = value
    except Exception as e:
        logger.error(f"Error setting session state for key {key}: {e}")

# ============================================================================
# UNSPLASH PHOTO SEARCH INTEGRATION
# ============================================================================

def search_stock_photos(search_terms: str, count: int = 3) -> List[Dict]:
    """Search for stock photos using Unsplash API"""
    try:
        UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")
        
        if not UNSPLASH_ACCESS_KEY:
            return [{
                "url": "https://via.placeholder.com/400x300/667eea/ffffff?text=Demo+Photo",
                "download_url": "https://via.placeholder.com/400x300/667eea/ffffff?text=Demo+Photo",
                "description": f"Demo photo for {search_terms}",
                "photographer": "Demo Photographer",
                "photographer_url": "#",
                "platform_recommendation": "Both Instagram and LinkedIn",
                "demo_mode": True
            }]
        
        headers = {
            "Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"
        }
        
        url = "https://api.unsplash.com/search/photos"
        params = {
            "query": search_terms,
            "per_page": count,
            "orientation": "landscape"
        }
        
        response = requests.get(url, headers=headers, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            photos = []
            
            for photo in data['results']:
                # Determine platform recommendation based on photo characteristics
                description = photo.get('alt_description', '').lower()
                platform_rec = "Both Instagram and LinkedIn"
                
                if any(word in description for word in ["abstract", "colorful", "vibrant", "creative"]):
                    platform_rec = "Instagram (more visual)"
                elif any(word in description for word in ["professional", "business", "office", "corporate"]):
                    platform_rec = "LinkedIn (more professional)"
                
                photo_info = {
                    "url": photo['urls']['regular'],
                    "download_url": photo['urls']['small'],
                    "description": photo.get('alt_description', 'Stock photo'),
                    "photographer": photo['user']['name'],
                    "photographer_url": photo['user']['links']['html'],
                    "platform_recommendation": platform_rec,
                    "demo_mode": False
                }
                photos.append(photo_info)
            
            return photos
        else:
            logger.error(f"Unsplash API error: {response.status_code}")
            return []
            
    except requests.exceptions.Timeout:
        logger.error("Unsplash API request timed out")
        return []
    except Exception as e:
        logger.error(f"Error searching photos: {e}")
        return []

def extract_topics_from_sheets_data(sheets_data: str) -> List[Dict]:
    """Extract topics and keywords from sheets data for photo search"""
    topics = []
    if not sheets_data:
        return topics
    
    lines = sheets_data.split('\n')
    current_topic = None
    current_keywords = None
    
    for line in lines:
        line = line.strip()
        if line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')):
            if 'Topic:' in line:
                current_topic = line.split('Topic:', 1)[1].strip()
        elif line.startswith('Keywords:') and current_topic:
            current_keywords = line.split('Keywords:', 1)[1].strip()
            
            # Extract search terms (first 3 keywords)
            search_terms = [term.strip() for term in current_keywords.split(',')[:3]]
            
            topics.append({
                'name': current_topic,
                'keywords': current_keywords,
                'search_terms': search_terms
            })
            current_topic = None
            current_keywords = None
    
    return topics

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def get_photos_for_topics(sheets_data: str) -> Dict[str, List[Dict]]:
    """Get photos for all topics with caching"""
    topics = extract_topics_from_sheets_data(sheets_data)
    all_photos = {}
    
    for topic in topics:
        topic_name = topic['name']
        search_terms = ' '.join(topic['search_terms'][:2])  # Use first 2 search terms
        
        photos = search_stock_photos(search_terms, 3)
        all_photos[topic_name] = photos
    
    return all_photos

# ============================================================================
# DATA VALIDATION
# ============================================================================

def validate_google_sheets_url(url: str) -> Tuple[bool, str]:
    """Validate Google Sheets URL format"""
    if not url:
        return False, "URL cannot be empty"
    
    if not url.startswith("https://docs.google.com/spreadsheets/"):
        return False, "Invalid Google Sheets URL format"
    
    if "/d/" not in url:
        return False, "Missing document ID in URL"
    
    return True, "Valid URL"

def validate_topic_keyword_pair(topic: str, keywords: str) -> Tuple[bool, str]:
    """Validate topic and keyword input"""
    if not topic or not topic.strip():
        return False, "Topic cannot be empty"
    
    if not keywords or not keywords.strip():
        return False, "Keywords cannot be empty"
    
    if len(topic) < 3:
        return False, "Topic must be at least 3 characters"
    
    if len(keywords) < 3:
        return False, "Keywords must be at least 3 characters"
    
    return True, "Valid input"

# ============================================================================
# GOOGLE SHEETS INTEGRATION
# ============================================================================

@st.cache_data(ttl=300)  # Cache for 5 minutes
def read_google_sheets_cached(sheet_url: str) -> Tuple[Optional[pd.DataFrame], str]:
    """Cached version of Google Sheets reading"""
    try:
        # Validate URL first
        is_valid, error_msg = validate_google_sheets_url(sheet_url)
        if not is_valid:
            return None, error_msg
        
        # Extract sheet ID
        if '/edit' in sheet_url:
            sheet_id = sheet_url.split('/d/')[1].split('/')[0]
        else:
            sheet_id = sheet_url.split('/d/')[1]
        
        csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
        
        # Read CSV data
        df = pd.read_csv(csv_url)
        
        if df.empty:
            return None, "No data found in the Google Sheet."
        
        # Validate required columns
        required_columns = ['topic', 'keywords']
        available_columns = [col.lower() for col in df.columns]
        
        missing_columns = [col for col in required_columns if col not in available_columns]
        if missing_columns:
            return None, f"Missing required columns: {', '.join(missing_columns)}"
        
        # Process data
        result = "Topics and Keywords from Google Sheets:\n\n"
        valid_rows = 0
        
        for i, row in df.iterrows():
            topic = row.get('topic', row.get('Topic', row.get('TOPIC', '')))
            keywords = row.get('keywords', row.get('Keywords', row.get('KEYWORDS', '')))
            
            is_valid, _ = validate_topic_keyword_pair(topic, keywords)
            if is_valid:
                result += f"{valid_rows+1}. Topic: {topic}\nKeywords: {keywords}\n\n"
                valid_rows += 1
        
        if valid_rows == 0:
            return None, "No valid topic and keyword pairs found."
        
        return df, result
        
    except pd.errors.EmptyDataError:
        return None, "The Google Sheet is empty."
    except pd.errors.ParserError:
        return None, "Unable to parse the Google Sheet. Please check the format."
    except requests.exceptions.Timeout:
        return None, "Request timed out. Please check your internet connection."
    except requests.exceptions.RequestException as e:
        return None, f"Network error: {str(e)}"
    except Exception as e:
        logger.error(f"Error reading Google Sheets: {e}")
        return None, f"Unexpected error: {str(e)}"

# ============================================================================
# BUFFER API INTEGRATION
# ============================================================================

class BufferAPI:
    def __init__(self):
        self.access_token = os.getenv("BUFFER_ACCESS_TOKEN")
        self.base_url = "https://api.bufferapp.com/1"
        self.session = requests.Session()
        self.session.timeout = 30
        
    def get_profiles(self) -> Dict:
        """Get all social media profiles connected to Buffer"""
        try:
            if not self.access_token:
                return {
                    "status": "demo",
                    "message": "DEMO MODE: Buffer API token not found. Add BUFFER_ACCESS_TOKEN to .env"
                }
            
            headers = {"Authorization": f"Bearer {self.access_token}"}
            response = self.session.get(f"{self.base_url}/profiles.json", headers=headers)
            
            if response.status_code == 200:
                return {
                    "status": "success",
                    "data": response.json()
                }
            else:
                return {
                    "status": "error",
                    "message": f"Error getting profiles: {response.status_code}"
                }
        except requests.exceptions.Timeout:
            return {
                "status": "error",
                "message": "Request timed out"
            }
        except Exception as e:
            logger.error(f"Buffer API error: {e}")
            return {
                "status": "error",
                "message": f"Error: {str(e)}"
            }
    
    def create_post(self, profile_ids: List[str], text: str, media_url: Optional[str] = None, scheduled_at: Optional[str] = None) -> Dict:
        """Create a post on Buffer with enhanced error handling"""
        try:
            if not self.access_token:
                # Simulate success for demo mode
                post_id = f"demo_post_{int(time.time())}"
                scheduled_time = scheduled_at if scheduled_at else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                return {
                    "success": True,
                    "id": post_id,
                    "text": text,
                    "profile_ids": profile_ids,
                    "scheduled_at": scheduled_time,
                    "media": {"link": media_url} if media_url else None,
                    "status": "scheduled" if scheduled_at else "posted",
                    "message": f"✅ Demo: Would {'schedule' if scheduled_at else 'post'} to {len(profile_ids)} profile(s)",
                    "demo_mode": True
                }
            
            headers = {"Authorization": f"Bearer {self.access_token}"}
            
            data = {
                "text": text,
                "profile_ids[]": profile_ids,
                "now": not scheduled_at
            }
            
            if media_url:
                data["media"] = {"link": media_url}
            
            if scheduled_at:
                data["scheduled_at"] = scheduled_at
            
            response = self.session.post(f"{self.base_url}/updates/create.json", headers=headers, data=data)
            
            if response.status_code == 200:
                result = response.json()
                result["success"] = True
                return result
            else:
                return {
                    "success": False,
                    "error": f"API Error: {response.status_code} - {response.text}"
                }
                
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error": "Request timed out"
            }
        except Exception as e:
            logger.error(f"Error creating Buffer post: {e}")
            return {
                "success": False,
                "error": str(e)
            }

# ============================================================================
# CREWAI INTEGRATION
# ============================================================================

@st.cache_resource
def initialize_llm():
    """Initialize the Gemini LLM with error handling"""
    try:
        from crewai import LLM
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            st.error("❌ GOOGLE_API_KEY not found in environment variables")
            return None
        
        llm = LLM(
            model="gemini/gemini-1.5-flash",
            api_key=api_key,
            temperature=0.7,
            max_tokens=1500,
        )
        return llm
    except ImportError:
        st.error("❌ CrewAI not installed. Please run: `pip install crewai`")
        return None
    except Exception as e:
        logger.error(f"Error initializing LLM: {e}")
        st.error(f"❌ Error initializing LLM: {str(e)}")
        return None

# ============================================================================
# IMPROVED CONTENT PARSING AND GENERATION
# ============================================================================

def parse_generated_content(content: str) -> Tuple[List[Dict], List[Dict]]:
    """
    Improved content parsing with better logic to handle AI-generated content
    Returns: (instagram_posts, linkedin_posts)
    """
    instagram_posts = []
    linkedin_posts = []
    
    if not content or not content.strip():
        return instagram_posts, linkedin_posts
    
    try:
        # Clean and normalize content
        content = content.strip()
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        current_topic = "Generated Content"
        current_platform = None
        current_post_content = []
        
        # Improved patterns for platform detection
        instagram_patterns = [
            r'instagram\s*posts?:?',
            r'📸\s*instagram',
            r'ig\s*posts?:?',
            r'instagram\s*content',
            r'instagram\s*captions?:?'
        ]
        
        linkedin_patterns = [
            r'linkedin\s*posts?:?',
            r'💼\s*linkedin',
            r'li\s*posts?:?',
            r'linkedin\s*content',
            r'linkedin\s*captions?:?'
        ]
        
        topic_patterns = [
            r'topic:?\s*(.+)',
            r'🎯\s*topic:?\s*(.+)',
            r'---\s*topic:?\s*(.+)'
        ]
        
        for line in lines:
            line_lower = line.lower()
            
            # Check for topic headers
            topic_match = None
            for pattern in topic_patterns:
                match = re.search(pattern, line_lower)
                if match:
                    topic_match = match.group(1).strip()
                    break
            
            if topic_match:
                # Save previous post if exists
                if current_post_content and current_platform:
                    post_data = {
                        'topic': current_topic,
                        'content': '\n'.join(current_post_content).strip()
                    }
                    if current_platform == 'instagram':
                        instagram_posts.append(post_data)
                    elif current_platform == 'linkedin':
                        linkedin_posts.append(post_data)
                
                current_topic = topic_match
                current_post_content = []
                current_platform = None
                continue
            
            # Check for platform headers
            is_instagram = any(re.search(pattern, line_lower) for pattern in instagram_patterns)
            is_linkedin = any(re.search(pattern, line_lower) for pattern in linkedin_patterns)
            
            if is_instagram:
                # Save previous post if exists
                if current_post_content and current_platform:
                    post_data = {
                        'topic': current_topic,
                        'content': '\n'.join(current_post_content).strip()
                    }
                    if current_platform == 'instagram':
                        instagram_posts.append(post_data)
                    elif current_platform == 'linkedin':
                        linkedin_posts.append(post_data)
                
                current_platform = 'instagram'
                current_post_content = []
                continue
            
            elif is_linkedin:
                # Save previous post if exists
                if current_post_content and current_platform:
                    post_data = {
                        'topic': current_topic,
                        'content': '\n'.join(current_post_content).strip()
                    }
                    if current_platform == 'instagram':
                        instagram_posts.append(post_data)
                    elif current_platform == 'linkedin':
                        linkedin_posts.append(post_data)
                
                current_platform = 'linkedin'
                current_post_content = []
                continue
            
            # Check for numbered posts (1. 2. etc.)
            post_number_match = re.match(r'^(\d+)\.\s*(.+)', line)
            if post_number_match and current_platform:
                # Save previous post if exists
                if current_post_content:
                    post_data = {
                        'topic': current_topic,
                        'content': '\n'.join(current_post_content).strip()
                    }
                    if current_platform == 'instagram':
                        instagram_posts.append(post_data)
                    elif current_platform == 'linkedin':
                        linkedin_posts.append(post_data)
                
                # Start new post
                current_post_content = [post_number_match.group(2)]
                continue
            
            # Add content to current post
            if current_platform and line:
                # Skip separator lines
                if line in ['---', '___', '***']:
                    continue
                current_post_content.append(line)
        
        # Save last post
        if current_post_content and current_platform:
            post_data = {
                'topic': current_topic,
                'content': '\n'.join(current_post_content).strip()
            }
            if current_platform == 'instagram':
                instagram_posts.append(post_data)
            elif current_platform == 'linkedin':
                linkedin_posts.append(post_data)
        
        # Fallback: If no platform-specific content found, try to split content intelligently
        if not instagram_posts and not linkedin_posts:
            # Look for emoji indicators
            instagram_lines = []
            linkedin_lines = []
            
            for line in lines:
                if any(emoji in line for emoji in ['📸', '🎯', '🚀', '💫', '✨', '#']):
                    instagram_lines.append(line)
                elif any(emoji in line for emoji in ['💼', '🏢', '📊', '💡']):
                    linkedin_lines.append(line)
                else:
                    # Split remaining content between platforms
                    if len(instagram_lines) <= len(linkedin_lines):
                        instagram_lines.append(line)
                    else:
                        linkedin_lines.append(line)
            
            if instagram_lines:
                instagram_posts.append({
                    'topic': current_topic,
                    'content': '\n'.join(instagram_lines)
                })
            
            if linkedin_lines:
                linkedin_posts.append({
                    'topic': current_topic,
                    'content': '\n'.join(linkedin_lines)
                })
    
    except Exception as e:
        logger.error(f"Error parsing content: {e}")
        # Fallback: Return raw content split
        content_lines = content.split('\n')
        mid_point = len(content_lines) // 2
        
        instagram_posts = [{
            'topic': 'Generated Content',
            'content': '\n'.join(content_lines[:mid_point])
        }]
        linkedin_posts = [{
            'topic': 'Generated Content', 
            'content': '\n'.join(content_lines[mid_point:])
        }]
    
    return instagram_posts, linkedin_posts

def render_photo_gallery(photos: List[Dict], topic: str):
    """Render photo gallery for a topic"""
    if not photos:
        st.warning("📷 No photos found for this topic")
        return
    
    st.markdown(f"**📸 Recommended Photos for '{topic}':**")
    
    # Display photos in columns
    cols = st.columns(min(len(photos), 3))
    
    for i, photo in enumerate(photos[:3]):
        with cols[i % 3]:
            # Photo display
            st.image(photo['download_url'])
            
            # Photo details with theme-compatible styling
            st.markdown(f"""
            <div style="
                background-color: var(--background-color);
                border: 1px solid var(--border-color);
                border-radius: 8px;
                padding: 0.75rem;
                margin-top: 0.5rem;
                color: var(--text-color);
            ">
                <div style="font-size: 0.8rem; margin-bottom: 0.5rem;">
                    <strong>Platform:</strong> {photo.get('platform_recommendation', 'Both')}
                </div>
                <div style="font-size: 0.8rem; margin-bottom: 0.5rem;">
                    <strong>Description:</strong> {photo.get('description', 'Stock photo')}
                </div>
                <div style="font-size: 0.75rem;">
                    📷 by <a href="{photo.get('photographer_url', '#')}" target="_blank">{photo.get('photographer', 'Unknown')}</a>
                    {' (Demo)' if photo.get('demo_mode') else ''}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Copy URL button
            if st.button(f"📋 Copy URL", key=f"copy_photo_{topic}_{i}"):
                st.code(photo['url'], language=None)
                st.success("📋 URL ready to copy!")

def render_post_card_with_photos(post: Dict, platform: str, post_number: int, photos: List[Dict] = None):
    """Render individual post card with photos - dark mode compatible"""
    
    if platform == "instagram":
        badge_icon = "📸"
        badge_text = "Instagram"
        platform_class = "instagram-card"
    else:  # linkedin
        badge_icon = "💼"
        badge_text = "LinkedIn"
        platform_class = "linkedin-card"
    
    st.markdown(f"""
    <div class="post-card {platform_class}">
        <div class="post-card-content">
            <div class="post-header">
                <span class="platform-badge {platform}-badge">
                    {badge_icon} {badge_text}
                </span>
                <span class="post-number">Post {post_number}</span>
            </div>
            <div class="topic-info">
                <strong>🎯 Topic:</strong> {post['topic']}
            </div>
            <div class="content-area">
                <div class="content-label">Content</div>
                <div class="content-text">{post['content']}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Show photos for this topic if available
    if photos:
        st.markdown("##### 📸 Recommended Photos:")
        render_photo_gallery(photos, post['topic'])

def create_content_crew_improved(sheets_data: str, config: Dict) -> Optional[str]:
    """Improved content generation with better prompting"""
    try:
        from crewai import Agent, Task, Crew, Process
        
        llm = initialize_llm()
        if not llm:
            return None
        
        # Enhanced content creator agent with clearer instructions
        content_creator = Agent(
            llm=llm,
            role="Expert Social Media Content Creator",
            goal=f"Generate exactly {config['num_instagram']} Instagram posts and {config['num_linkedin']} LinkedIn posts for each topic with {config['tone'].lower()} tone.",
            backstory="""You are a seasoned social media strategist with 10+ years of experience creating viral content. 
            You understand platform-specific audiences: Instagram users love visual, engaging content with emojis and hashtags, 
            while LinkedIn users prefer professional insights with thought-provoking questions.""",
            verbose=False
        )
        
        # Improved task with very specific formatting requirements
        caption_task = Task(
            description=f"""
            Create social media content using this data:
            {sheets_data}
            
            REQUIREMENTS:
            - Tone: {config['tone']}
            - Include hashtags: {config['include_hashtags']}
            - Include emojis: {config['include_emojis']}
            - Generate {config['num_instagram']} Instagram posts per topic
            - Generate {config['num_linkedin']} LinkedIn posts per topic
            
            CRITICAL: Follow this EXACT format for each topic:
            
            TOPIC: [Topic Name]
            
            INSTAGRAM CAPTIONS:
            1. [First Instagram post with {'hashtags and ' if config['include_hashtags'] else ''}{'emojis and ' if config['include_emojis'] else ''}call-to-action]
            {'2. [Second Instagram post...]' if config['num_instagram'] > 1 else ''}
            {'3. [Third Instagram post...]' if config['num_instagram'] > 2 else ''}
            
            LINKEDIN CAPTIONS:
            1. [First LinkedIn post with professional insights and engagement question]
            {'2. [Second LinkedIn post...]' if config['num_linkedin'] > 1 else ''}
            {'3. [Third LinkedIn post...]' if config['num_linkedin'] > 2 else ''}
            
            ---
            
            [Repeat for next topic]
            """,
            expected_output=f"""
            Properly formatted social media content with clear platform sections.
            Each topic should have exactly {config['num_instagram']} Instagram posts and {config['num_linkedin']} LinkedIn posts.
            Use the EXACT format specified with "TOPIC:", "INSTAGRAM CAPTIONS:", and "LINKEDIN CAPTIONS:" headers.
            All content should be in {config['tone'].lower()} tone and incorporate the provided keywords naturally.
            """,
            agent=content_creator
        )
        
        # Run crew
        crew = Crew(
            agents=[content_creator],
            tasks=[caption_task],
            process=Process.sequential,
            verbose=False
        )
        
        result = crew.kickoff()
        
        # Extract result properly
        if hasattr(result, 'raw'):
            return result.raw
        elif hasattr(result, 'result'):
            return result.result
        elif hasattr(result, 'output'):
            return result.output
        else:
            return str(result)
        
    except Exception as e:
        logger.error(f"Error in improved content generation: {e}")
        raise e

def render_generated_content_preview_with_photos(content: str, context: str = "default"):
    """Enhanced content preview with photos - dark mode compatible"""
    if not content:
        st.warning("No content to display")
        return
    
    # Parse content using improved logic
    instagram_posts, linkedin_posts = parse_generated_content(content)
    
    # Remove duplicates based on content
    seen_instagram = set()
    unique_instagram = []
    for post in instagram_posts:
        content_hash = hash_string(post['content'])
        if content_hash not in seen_instagram:
            seen_instagram.add(content_hash)
            unique_instagram.append(post)
    
    seen_linkedin = set()
    unique_linkedin = []
    for post in linkedin_posts:
        content_hash = hash_string(post['content'])
        if content_hash not in seen_linkedin:
            seen_linkedin.add(content_hash)
            unique_linkedin.append(post)
    
    instagram_posts = unique_instagram
    linkedin_posts = unique_linkedin
    
    # Get photos for all topics
    sheets_data = safe_get_session_state('sheets_data', '')
    if sheets_data:
        with st.spinner("🔍 Searching for relevant photos..."):
            try:
                topic_photos = get_photos_for_topics(sheets_data)
                safe_set_session_state('topic_photos', topic_photos)
            except Exception as e:
                logger.error(f"Error getting photos: {e}")
                topic_photos = {}
    else:
        topic_photos = safe_get_session_state('topic_photos', {})
    
    # Display Instagram posts with photos
    if instagram_posts:
        st.markdown('<div class="platform-header instagram-header"><h2>📸 Instagram Posts</h2><p>Vibrant social media content for your Instagram audience</p></div>', unsafe_allow_html=True)
        
        for i, post in enumerate(instagram_posts, 1):
            photos = topic_photos.get(post['topic'], [])
            # Filter photos recommended for Instagram
            instagram_photos = [p for p in photos if 'Instagram' in p.get('platform_recommendation', '')]
            if not instagram_photos:
                instagram_photos = photos  # Use all photos if no specific Instagram ones
            
            render_post_card_with_photos(post, "instagram", i, instagram_photos)
    
    # Display LinkedIn posts with photos
    if linkedin_posts:
        st.markdown('<div class="platform-header linkedin-header"><h2>💼 LinkedIn Posts</h2><p>Professional content for your LinkedIn network</p></div>', unsafe_allow_html=True)
        
        for i, post in enumerate(linkedin_posts, 1):
            photos = topic_photos.get(post['topic'], [])
            # Filter photos recommended for LinkedIn
            linkedin_photos = [p for p in photos if 'LinkedIn' in p.get('platform_recommendation', '')]
            if not linkedin_photos:
                linkedin_photos = photos  # Use all photos if no specific LinkedIn ones
            
            render_post_card_with_photos(post, "linkedin", i, linkedin_photos)
    
    # Show message if no posts found
    if not instagram_posts and not linkedin_posts:
        st.warning("⚠️ No posts could be parsed from the generated content. The AI may have generated content in an unexpected format.")
        
        # Show raw content as fallback
        with st.expander("🔍 Debug: View Raw Content", expanded=True):
            st.text_area("Raw AI Output:", content, height=300, key=f"debug_raw_{context}")
    
    # Photo summary section
    if topic_photos:
        with st.expander("📸 Photo Summary for All Topics", expanded=False):
            total_photos = sum(len(photos) for photos in topic_photos.values())
            st.info(f"📊 Found {total_photos} photos across {len(topic_photos)} topics")
            
            for topic, photos in topic_photos.items():
                st.write(f"**{topic}:** {len(photos)} photos")
                if photos and not photos[0].get('demo_mode'):
                    st.write(f"   • {', '.join([p['photographer'] for p in photos[:3]])}")
    
    # Raw content option (always available)
    with st.expander("📄 View Raw Content", expanded=False):
        st.markdown("#### Raw Generated Content")
        unique_key = f"raw_content_{context}_{hash_string(content)[:8]}_{int(time.time())}"
        st.text_area("", content, height=300, key=unique_key)

# ============================================================================
# UI COMPONENTS WITH DARK MODE SUPPORT
# ============================================================================

def render_header():
    """Render the main header with dark mode compatible styling"""
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* CSS Variables for theme compatibility */
    :root {
        --primary-color: #667eea;
        --secondary-color: #764ba2;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --error-color: #ef4444;
        --instagram-color: #ff6b6b;
        --linkedin-color: #667eea;
    }
    
    /* Dark mode CSS variables - Streamlit automatically applies these */
    [data-theme="dark"] {
        --text-color: #ffffff;
        --background-color: #0e1117;
        --secondary-background: #262730;
        --border-color: #30363d;
        --card-background: #1a1d24;
    }
    
    /* Light mode CSS variables */
    [data-theme="light"], :root {
        --text-color: #1e293b;
        --background-color: #ffffff;
        --secondary-background: #f8fafc;
        --border-color: #e2e8f0;
        --card-background: #ffffff;
    }
    
    /* Main header styling */
    .main-header {
        text-align: center;
        padding: 3rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        border-radius: 20px;
        margin: 1rem 0;
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="25" cy="25" r="1" fill="white" opacity="0.1"/><circle cx="75" cy="75" r="1" fill="white" opacity="0.1"/><circle cx="50" cy="10" r="0.5" fill="white" opacity="0.1"/><circle cx="10" cy="60" r="0.5" fill="white" opacity="0.1"/><circle cx="90" cy="40" r="0.5" fill="white" opacity="0.1"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>');
        opacity: 0.3;
    }
    
    .main-title {
        font-family: 'Inter', sans-serif;
        font-size: 4rem;
        font-weight: 800;
        color: white;
        margin-bottom: 1rem;
        text-shadow: 0 4px 8px rgba(0,0,0,0.3);
        position: relative;
        z-index: 2;
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { text-shadow: 0 4px 8px rgba(0,0,0,0.3), 0 0 20px rgba(255,255,255,0.5); }
        to { text-shadow: 0 4px 8px rgba(0,0,0,0.3), 0 0 30px rgba(255,255,255,0.8); }
    }
    
    .main-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.3rem;
        color: rgba(255,255,255,0.9);
        margin-bottom: 0;
        font-weight: 300;
        position: relative;
        z-index: 2;
    }
    
    .floating-icons {
        position: absolute;
        top: 20px;
        right: 30px;
        font-size: 2rem;
        animation: float 3s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    /* Enhanced metrics with theme support */
    .enhanced-metric {
        background-color: var(--card-background);
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid var(--border-color);
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
        color: var(--text-color);
    }
    
    .enhanced-metric:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.1);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--text-color);
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: var(--text-color);
        opacity: 0.7;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 2s infinite;
    }
    
    .status-success { background-color: var(--success-color); }
    .status-warning { background-color: var(--warning-color); }
    .status-error { background-color: var(--error-color); }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    /* Platform headers */
    .platform-header {
        text-align: center;
        padding: 2rem;
        border-radius: 20px;
        margin: 2rem 0;
        position: relative;
        overflow: hidden;
    }
    
    .instagram-header {
        background: linear-gradient(135deg, #ff6b6b 0%, #feca57 25%, #48dbfb 50%, #ff9ff3 75%, #54a0ff 100%);
    }
    
    .linkedin-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #f5576c 75%, #4facfe 100%);
    }
    
    .platform-header h2 {
        color: white;
        margin-bottom: 0.5rem;
        font-size: 2.5rem;
        font-weight: 800;
        text-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    
    .platform-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        margin: 0;
    }
    
    /* Post cards with theme support */
    .post-card {
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
        transition: all 0.4s ease;
    }
    
    .instagram-card {
        background: linear-gradient(135deg, #ff6b6b 0%, #feca57 25%, #48dbfb 50%, #ff9ff3 75%, #54a0ff 100%);
        border: 3px solid var(--instagram-color);
        box-shadow: 0 12px 30px rgba(255, 107, 107, 0.3);
    }
    
    .linkedin-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #f5576c 75%, #4facfe 100%);
        border: 3px solid var(--linkedin-color);
        box-shadow: 0 12px 30px rgba(102, 126, 234, 0.3);
    }
    
    .post-card-content {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 16px;
        padding: 1.5rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        color: #1e293b;
    }
    
    .post-header {
        display: flex;
        align-items: center;
        margin-bottom: 1rem;
    }
    
    .platform-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: white;
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.4);
        animation: pulse 2s infinite;
    }
    
    .instagram-badge {
        background: linear-gradient(135deg, #ff6b6b, #feca57);
    }
    
    .linkedin-badge {
        background: linear-gradient(135deg, #667eea, #764ba2);
    }
    
    .post-number {
        margin-left: auto;
        font-weight: 700;
        color: var(--primary-color);
        font-size: 1.2rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .topic-info {
        color: #2d3748;
        margin-bottom: 0.5rem;
        font-size: 0.95rem;
        font-weight: 600;
    }
    
    .content-area {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 2px solid #cbd5e1;
        line-height: 1.7;
        margin-top: 1rem;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.05);
        position: relative;
        color: #2d3748;
        white-space: pre-wrap;
    }
    
    .content-label {
        position: absolute;
        top: -8px;
        left: 20px;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    
    .content-text {
        margin-top: 0.5rem;
    }
    
    /* Enhanced cards with theme support */
    .enhanced-card {
        background-color: var(--card-background);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid var(--border-color);
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
        margin-bottom: 1rem;
        color: var(--text-color);
    }
    
    .enhanced-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    
    /* Notification boxes with theme support */
    .notification-box {
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-width: 1px;
        border-style: solid;
    }
    
    .notification-success {
        background-color: rgba(16, 185, 129, 0.1);
        border-color: var(--success-color);
        color: var(--text-color);
    }
    
    .notification-warning {
        background-color: rgba(245, 158, 11, 0.1);
        border-color: var(--warning-color);
        color: var(--text-color);
    }
    
    .notification-error {
        background-color: rgba(239, 68, 68, 0.1);
        border-color: var(--error-color);
        color: var(--text-color);
    }
    
    .notification-info {
        background-color: rgba(59, 130, 246, 0.1);
        border-color: #3b82f6;
        color: var(--text-color);
    }
    
    .notification-title {
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .notification-content {
        margin: 0;
    }
    
    /* Remove forced colors and use theme-aware colors */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        color: var(--text-color) !important;
    }
    
    .stMarkdown p {
        color: var(--text-color) !important;
    }
    
    /* Button styling that respects theme */
    .stButton > button {
        background-color: var(--primary-color) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        background-color: #5a67d8 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3) !important;
    }
    
    /* Form elements that adapt to theme */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > div {
        background-color: var(--background-color) !important;
        color: var(--text-color) !important;
        border-color: var(--border-color) !important;
    }
    
    /* Checkbox and radio button styling */
    .stCheckbox > label,
    .stRadio > div > label {
        color: var(--text-color) !important;
    }
    
    /* Metric styling */
    [data-testid="metric-container"] {
        background-color: var(--card-background);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1rem;
    }
    
    [data-testid="metric-container"] > div {
        color: var(--text-color) !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        color: var(--text-color) !important;
        background-color: var(--card-background) !important;
        border-color: var(--border-color) !important;
    }
    
    /* Dataframe styling */
    .dataframe {
        background-color: var(--card-background) !important;
        color: var(--text-color) !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: var(--background-color);
    }
    
    .stTabs [data-baseweb="tab"] {
        color: var(--text-color) !important;
    }
    
    /* Info/warning/error boxes */
    .stAlert {
        background-color: var(--card-background) !important;
        color: var(--text-color) !important;
        border-color: var(--border-color) !important;
    }
    
    /* Download button styling */
    .stDownloadButton > button {
        background-color: var(--secondary-color) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
    }
    
    /* Code block styling */
    .stCode {
        background-color: var(--secondary-background) !important;
        color: var(--text-color) !important;
        border-color: var(--border-color) !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: var(--background-color) !important;
    }
    
    /* Animation for rainbow effect */
    @keyframes rainbow {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    </style>
    
    <div class="main-header">
        <div class="floating-icons">🚀</div>
        <h1 class="main-title">Social Media Content Agent</h1>
        <p class="main-subtitle">AI-powered content generation with stock photo recommendations</p>
    </div>
    """, unsafe_allow_html=True)

def render_status_metrics():
    """Render status metrics with theme-compatible styling"""
    api_status = validate_api_keys()
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        content_count = safe_get_session_state('content_generated_count', 0)
        st.markdown(f"""
        <div class="enhanced-metric">
            <div class="metric-value">{content_count}</div>
            <div class="metric-label">Content Generated</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        posts_count = len(safe_get_session_state('posting_results', []))
        st.markdown(f"""
        <div class="enhanced-metric">
            <div class="metric-value">{posts_count}</div>
            <div class="metric-label">Posts Created</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        topic_photos = safe_get_session_state('topic_photos', {})
        photo_count = sum(len(photos) for photos in topic_photos.values())
        st.markdown(f"""
        <div class="enhanced-metric">
            <div class="metric-value">{photo_count}</div>
            <div class="metric-label">Photos Found</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        gemini_status = "✅" if api_status["gemini"] else "❌"
        status_class = "status-success" if api_status["gemini"] else "status-error"
        st.markdown(f"""
        <div class="enhanced-metric">
            <div class="metric-value">
                <span class="status-indicator {status_class}"></span>
                {gemini_status}
            </div>
            <div class="metric-label">Gemini API</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        unsplash_status = "✅" if api_status["unsplash"] else "⚠️"
        status_class = "status-success" if api_status["unsplash"] else "status-warning"
        st.markdown(f"""
        <div class="enhanced-metric">
            <div class="metric-value">
                <span class="status-indicator {status_class}"></span>
                {unsplash_status}
            </div>
            <div class="metric-label">Unsplash API</div>
        </div>
        """, unsafe_allow_html=True)

def render_content_generation_tab():
    """Enhanced content generation tab with photo integration"""
    st.markdown("### 🎯 Generate Social Media Content with Photos")
    
    # Check if data is available
    sheets_data = safe_get_session_state('sheets_data', None)
    if not sheets_data:
        st.markdown("""
        <div class="notification-box notification-warning">
            <div class="notification-title">⚠️ No Data Available</div>
            <div class="notification-content">Please add data in the 'Data Source' tab first.</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="notification-box notification-info">
            <div class="notification-title">💡 Quick Start</div>
            <div class="notification-content">Go to the 'Data Source' tab and click 'Load Sample Data' to test the app!</div>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Show current data preview
    with st.expander("📊 Current Data Preview", expanded=False):
        st.text_area("Your topics and keywords:", sheets_data, height=150, disabled=True)
    
    # Configuration form
    st.markdown('<div class="enhanced-card"><h3>📋 Content & Photo Settings</h3></div>', unsafe_allow_html=True)
    
    with st.form("improved_content_config"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Instagram Captions per Topic**")
            num_instagram = st.slider("", 1, 5, 1, key="instagram_slider")
            st.markdown("**LinkedIn Captions per Topic**")
            num_linkedin = st.slider("", 1, 5, 1, key="linkedin_slider")
            
        with col2:
            st.markdown("**Content Tone**")
            tone = st.selectbox("", ["Professional", "Casual", "Friendly", "Authoritative"], key="tone_select")
            col_a, col_b = st.columns(2)
            with col_a:
                include_hashtags = st.checkbox("Include Hashtags", value=True, key="hashtags_check")
            with col_b:
                include_emojis = st.checkbox("Include Emojis", value=True, key="emojis_check")
        
        # Photo settings
        st.markdown("**📸 Photo Settings**")
        col_c, col_d = st.columns(2)
        with col_c:
            include_photos = st.checkbox("Search for stock photos", value=True, key="photos_check")
        with col_d:
            if include_photos:
                photos_per_topic = st.slider("Photos per topic", 1, 5, 3, key="photos_slider")
            else:
                photos_per_topic = 0
        
        # Submit button
        submitted = st.form_submit_button("🚀 Generate Content & Photos", type="primary", use_container_width=True)
        
        if submitted:
            config = {
                'num_instagram': num_instagram,
                'num_linkedin': num_linkedin,
                'tone': tone,
                'include_hashtags': include_hashtags,
                'include_emojis': include_emojis,
                'include_photos': include_photos,
                'photos_per_topic': photos_per_topic
            }
            
            # Show generation progress
            progress_placeholder = st.empty()
            progress_placeholder.markdown("""
            <div class="notification-box notification-info">
                <div class="notification-title">🤖 AI is generating your content...</div>
            </div>
            """, unsafe_allow_html=True)
            
            try:
                # Generate content
                result = create_content_crew_improved(sheets_data, config)
                
                if result:
                    safe_set_session_state('generated_content', result)
                    safe_set_session_state('content_generated', True)
                    safe_set_session_state('content_generated_count', 
                                         safe_get_session_state('content_generated_count', 0) + 1)
                    
                    # Generate photos if requested
                    if include_photos:
                        progress_placeholder.markdown("""
                        <div class="notification-box notification-info">
                            <div class="notification-title">📸 Searching for relevant photos...</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        try:
                            topic_photos = get_photos_for_topics(sheets_data)
                            safe_set_session_state('topic_photos', topic_photos)
                            total_photos = sum(len(photos) for photos in topic_photos.values())
                            
                            progress_placeholder.markdown(f"""
                            <div class="notification-box notification-success">
                                <div class="notification-title">✅ Content and photos generated successfully!</div>
                                <div class="notification-content">📸 Found {total_photos} stock photos across {len(topic_photos)} topics</div>
                            </div>
                            """, unsafe_allow_html=True)
                        except Exception as photo_error:
                            logger.error(f"Photo search error: {photo_error}")
                            progress_placeholder.markdown("""
                            <div class="notification-box notification-success">
                                <div class="notification-title">✅ Content generated successfully!</div>
                                <div class="notification-content">⚠️ Photo search failed, but content is ready</div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        progress_placeholder.markdown("""
                        <div class="notification-box notification-success">
                            <div class="notification-title">✅ Content generated successfully!</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Show preview with enhanced UI
                    with st.expander("📝 Preview Generated Content & Photos", expanded=True):
                        render_generated_content_preview_with_photos(result, "generation")
                    
                    # Download options
                    col_dl1, col_dl2 = st.columns(2)
                    with col_dl1:
                        st.download_button(
                            label="📥 Download Content",
                            data=result,
                            file_name=f"social_content_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain",
                            type="secondary"
                        )
                    
                    with col_dl2:
                        if include_photos and safe_get_session_state('topic_photos'):
                            # Create comprehensive output with photos
                            comprehensive_output = create_comprehensive_output_with_photos(
                                result, 
                                safe_get_session_state('topic_photos', {}),
                                sheets_data
                            )
                            st.download_button(
                                label="📥 Download with Photos",
                                data=comprehensive_output,
                                file_name=f"content_with_photos_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                mime="text/plain",
                                type="secondary"
                            )
                else:
                    progress_placeholder.empty()
                    st.markdown("""
                    <div class="notification-box notification-error">
                        <div class="notification-title">❌ Failed to generate content. Please try again.</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
            except Exception as e:
                progress_placeholder.empty()
                st.markdown(f"""
                <div class="notification-box notification-error">
                    <div class="notification-title">❌ Error generating content</div>
                    <div class="notification-content">{str(e)}</div>
                </div>
                """, unsafe_allow_html=True)
                logger.error(f"Content generation error: {e}")
                
                # Show debug info
                with st.expander("🔧 Debug Information"):
                    st.write(f"**Error:** {str(e)}")
                    st.write(f"**Config:** {config}")
                    st.write(f"**Data length:** {len(sheets_data)} characters")

def create_comprehensive_output_with_photos(content: str, topic_photos: Dict, sheets_data: str) -> str:
    """Create comprehensive output including content and photos"""
    output = "INTEGRATED SOCIAL MEDIA CONTENT & STOCK PHOTOS\n"
    output += "=" * 60 + "\n\n"
    output += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    # Add social media content
    output += "📝 SOCIAL MEDIA CONTENT:\n"
    output += "-" * 30 + "\n"
    output += content + "\n\n"
    
    # Add photo recommendations
    output += "📸 STOCK PHOTO RECOMMENDATIONS:\n"
    output += "-" * 35 + "\n\n"
    
    for topic, photos in topic_photos.items():
        output += f"TOPIC: {topic}\n"
        output += "-" * 40 + "\n\n"
        
        if photos:
            for i, photo in enumerate(photos, 1):
                output += f"{i}. Photo URL: {photo['url']}\n"
                output += f"   Description: {photo['description']}\n"
                output += f"   Best for: {photo['platform_recommendation']}\n"
                output += f"   Photographer: {photo['photographer']}\n"
                if not photo.get('demo_mode'):
                    output += f"   Attribution: Photo by {photo['photographer']} on Unsplash\n"
                else:
                    output += f"   Note: Demo photo - replace with actual Unsplash photo\n"
                output += f"   Photo Link: {photo['photographer_url']}\n\n"
        else:
            output += "   No photos found for this topic\n\n"
    
    return output

def render_data_source_tab():
    """Render the data source tab with dark mode compatible styling"""
    st.markdown("### 📊 Data Source Configuration")
    
    # Data input method selection
    st.markdown('<div class="enhanced-card"><h3>Choose Data Source</h3></div>', unsafe_allow_html=True)
    
    data_method = st.radio(
        "Choose data source:",
        ["Google Sheets URL", "Manual Input", "Sample Data"],
        horizontal=True
    )
    
    if data_method == "Google Sheets URL":
        st.markdown("""
        <div class="notification-box notification-info">
            <div class="notification-title">🔗 Google Sheets Integration</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("**Google Sheets URL:**")
        st.caption("Make sure your sheet has 'Topic' and 'Keywords' columns")
        sheet_url = st.text_input(
            "",
            placeholder="https://docs.google.com/spreadsheets/d/...",
            key="sheets_url_input"
        )
        
        if st.button("📥 Load from Google Sheets", type="primary", use_container_width=True):
            if sheet_url:
                with st.spinner("Loading data from Google Sheets..."):
                    df, result = read_google_sheets_cached(sheet_url)
                    
                    if df is not None:
                        safe_set_session_state('sheets_data', result)
                        safe_set_session_state('dataframe', df)
                        st.markdown("""
                        <div class="notification-box notification-success">
                            <div class="notification-title">✅ Data loaded successfully!</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Display data
                        st.markdown('<div class="enhanced-card"><h4>📋 Loaded Data</h4></div>', unsafe_allow_html=True)
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.markdown(f"""
                        <div class="notification-box notification-error">
                            <div class="notification-title">❌ Error</div>
                            <div class="notification-content">{result}</div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="notification-box notification-warning">
                    <div class="notification-title">⚠️ Please enter a Google Sheets URL</div>
                </div>
                """, unsafe_allow_html=True)
    
    elif data_method == "Manual Input":
        st.markdown("""
        <div class="notification-box notification-warning">
            <div class="notification-title">✏️ Manual Data Entry</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Manual input form
        with st.form("manual_data"):
            st.markdown('<div class="enhanced-card"><h4>Enter Your Topics and Keywords</h4></div>', unsafe_allow_html=True)
            
            topics = []
            keywords = []
            
            for i in range(3):  # Allow up to 3 topics
                st.markdown(f"**Topic {i+1}**")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Topic**")
                    topic = st.text_input("", key=f"topic_{i}", placeholder="Enter topic...")
                with col2:
                    st.markdown("**Keywords**")
                    keyword = st.text_input("", key=f"keyword_{i}", placeholder="Enter keywords...")
                
                if topic and keyword:
                    is_valid, error_msg = validate_topic_keyword_pair(topic, keyword)
                    if is_valid:
                        topics.append(topic)
                        keywords.append(keyword)
                    else:
                        st.markdown(f"""
                        <div class="notification-box notification-error">
                            <div class="notification-content">Topic {i+1}: {error_msg}</div>
                        </div>
                        """, unsafe_allow_html=True)
            
            if st.form_submit_button("💾 Save Manual Data", type="primary", use_container_width=True):
                if topics:
                    # Create formatted data
                    result = "Topics and Keywords from Manual Input:\n\n"
                    for i, (topic, keyword) in enumerate(zip(topics, keywords)):
                        result += f"{i+1}. Topic: {topic}\nKeywords: {keyword}\n\n"
                    
                    safe_set_session_state('sheets_data', result)
                    st.markdown("""
                    <div class="notification-box notification-success">
                        <div class="notification-title">✅ Manual data saved!</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="notification-box notification-warning">
                        <div class="notification-title">⚠️ Please enter at least one valid topic and keywords</div>
                    </div>
                    """, unsafe_allow_html=True)
    
    else:  # Sample Data
        st.markdown("""
        <div class="notification-box notification-info">
            <div class="notification-title">📝 Sample Data</div>
            <div class="notification-content">Click the button below to load sample data for testing</div>
        </div>
        """, unsafe_allow_html=True)
        
        sample_data = """Topics and Keywords from Sample Data:

1. Topic: AI in healthcare
Keywords: AI, healthcare, medical technology, patient care

2. Topic: Remote work trends
Keywords: remote work, productivity, digital workspace, collaboration

3. Topic: E-commerce growth
Keywords: ecommerce, online shopping, digital marketing, customer experience

"""
        
        # Display sample data preview
        st.markdown('<div class="enhanced-card"><h4>Sample Data Preview</h4></div>', unsafe_allow_html=True)
        st.text_area("", sample_data, height=200, disabled=True, key="sample_preview")
        
        # Button to load sample data
        if st.button("📥 Load Sample Data", type="primary", use_container_width=True):
            safe_set_session_state('sheets_data', sample_data)
            st.markdown("""
            <div class="notification-box notification-success">
                <div class="notification-title">✅ Sample data loaded successfully!</div>
                <div class="notification-content">💡 Now go to the 'Generate Content' tab to create your social media posts with photos!</div>
            </div>
            """, unsafe_allow_html=True)

def render_posting_tab():
    """Enhanced posting tab with photo integration and dark mode support"""
    st.markdown("### 📱 Social Media Posting with Photos")
    
    if not safe_get_session_state('content_generated', False):
        st.warning("⚠️ Please generate content first in the 'Generate Content' tab")
        return
    
    # Get generated content and parse it
    generated_content = safe_get_session_state('generated_content', '')
    instagram_posts, linkedin_posts = parse_generated_content(generated_content)
    
    if not instagram_posts and not linkedin_posts:
        st.error("❌ No valid posts found in generated content")
        return
    
    # Get photos data
    topic_photos = safe_get_session_state('topic_photos', {})
    
    # Content selection with photos
    st.markdown("#### 📝 Select Content to Post")
    
    selected_posts = []
    
    # Instagram post selection with photo preview
    if instagram_posts:
        st.markdown("**📸 Instagram Posts:**")
        for i, post in enumerate(instagram_posts):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                selected = st.checkbox(f"Instagram Post {i+1}: {post['topic']}", key=f"ig_post_{i}")
                if selected:
                    # Get recommended photo for this post
                    photos = topic_photos.get(post['topic'], [])
                    instagram_photos = [p for p in photos if 'Instagram' in p.get('platform_recommendation', '')]
                    selected_photo = instagram_photos[0] if instagram_photos else (photos[0] if photos else None)
                    
                    selected_posts.append({
                        'platform': 'Instagram',
                        'content': post['content'],
                        'topic': post['topic'],
                        'recommended_photo': selected_photo
                    })
            
            with col2:
                if topic_photos.get(post['topic']):
                    photos = topic_photos[post['topic']]
                    instagram_photos = [p for p in photos if 'Instagram' in p.get('platform_recommendation', '')]
                    if not instagram_photos:
                        instagram_photos = photos[:1]  # Show first photo if no Instagram-specific ones
                    
                    if instagram_photos:
                        st.image(instagram_photos[0]['download_url'], width=200)
                        st.caption(f"📷 {instagram_photos[0]['description']}")
    
    # LinkedIn post selection with photo preview
    if linkedin_posts:
        st.markdown("**💼 LinkedIn Posts:**")
        for i, post in enumerate(linkedin_posts):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                selected = st.checkbox(f"LinkedIn Post {i+1}: {post['topic']}", key=f"li_post_{i}")
                if selected:
                    # Get recommended photo for this post
                    photos = topic_photos.get(post['topic'], [])
                    linkedin_photos = [p for p in photos if 'LinkedIn' in p.get('platform_recommendation', '')]
                    selected_photo = linkedin_photos[0] if linkedin_photos else (photos[0] if photos else None)
                    
                    selected_posts.append({
                        'platform': 'LinkedIn',
                        'content': post['content'],
                        'topic': post['topic'],
                        'recommended_photo': selected_photo
                    })
            
            with col2:
                if topic_photos.get(post['topic']):
                    photos = topic_photos[post['topic']]
                    linkedin_photos = [p for p in photos if 'LinkedIn' in p.get('platform_recommendation', '')]
                    if not linkedin_photos:
                        linkedin_photos = photos[:1]  # Show first photo if no LinkedIn-specific ones
                    
                    if linkedin_photos:
                        st.image(linkedin_photos[0]['download_url'], width=200)
                        st.caption(f"📷 {linkedin_photos[0]['description']}")
    
    if not selected_posts:
        st.info("👆 Select posts above to configure posting settings")
        return
    
    # Show selected posts summary
    st.markdown("#### 📋 Selected Posts Summary")
    for i, post in enumerate(selected_posts, 1):
        photo_info = "📸 With photo" if post.get('recommended_photo') else "📝 Text only"
        st.write(f"{i}. **{post['platform']}** - {post['topic']} ({photo_info})")
    
    # Posting configuration
    st.markdown("#### ⚙️ Posting Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📅 Scheduling**")
        post_now = st.checkbox("Post Now", value=False)
        
        if not post_now:
            schedule_hours = st.slider("Schedule in hours", 1, 24, 2)
            schedule_time = datetime.now() + timedelta(hours=schedule_hours)
            st.info(f"📅 Will post at: {schedule_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    with col2:
        st.markdown("**📸 Photo Settings**")
        auto_attach_photos = st.checkbox("Auto-attach recommended photos", value=True)
        if not auto_attach_photos:
            custom_media_url = st.text_input("Custom media URL (optional)", placeholder="https://example.com/image.jpg")
        else:
            custom_media_url = None
    
    # Buffer API status
    buffer = BufferAPI()
    buffer_status = buffer.get_profiles()
    
    if buffer_status["status"] == "demo":
        st.info(f"ℹ️ {buffer_status['message']}")
    elif buffer_status["status"] == "success":
        st.success("✅ Buffer API connected")
        # Show available profiles
        profiles = buffer_status.get("data", [])
        if profiles:
            st.write(f"**Connected profiles:** {len(profiles)}")
    else:
        st.error(f"❌ Buffer API error: {buffer_status['message']}")
    
    # Post button
    if st.button("📤 Post Selected Content with Photos", type="primary", use_container_width=True):
        if not selected_posts:
            st.error("❌ Please select at least one post")
            return
        
        with st.spinner(f"📤 Posting {len(selected_posts)} posts with photos..."):
            try:
                results = []
                
                for i, post_data in enumerate(selected_posts):
                    platform = post_data['platform']
                    content = post_data['content']
                    topic = post_data['topic']
                    recommended_photo = post_data.get('recommended_photo')
                    
                    # Determine media URL
                    media_url = None
                    if auto_attach_photos and recommended_photo:
                        media_url = recommended_photo['url']
                    elif custom_media_url:
                        media_url = custom_media_url
                    
                    # Create post
                    result = buffer.create_post(
                        profile_ids=[f"demo_{platform.lower()}_profile"],
                        text=content,
                        media_url=media_url,
                        scheduled_at=None if post_now else schedule_time.strftime("%Y-%m-%d %H:%M:%S")
                    )
                    
                    results.append({
                        "platform": platform,
                        "topic": topic,
                        "result": result,
                        "content": content[:100] + "..." if len(content) > 100 else content,
                        "media_url": media_url,
                        "photo_info": recommended_photo
                    })
                    
                    # Add small delay between posts
                    time.sleep(0.5)
                
                safe_set_session_state('posting_results', results)
                
                # Show results
                st.success(f"🎉 {len(selected_posts)} posts processed!")
                
                for result in results:
                    platform = result['platform']
                    topic = result['topic']
                    post_result = result['result']
                    media_url = result.get('media_url')
                    
                    photo_status = " with photo" if media_url else ""
                    
                    if post_result.get('success'):
                        st.success(f"✅ {platform} ({topic}){photo_status}: {post_result.get('message', 'Successfully posted')}")
                    else:
                        st.error(f"❌ {platform} ({topic}): {post_result.get('error', 'Unknown error')}")
                
            except Exception as e:
                st.error(f"❌ Error posting content: {str(e)}")
                logger.error(f"Posting error: {e}")

def render_results_tab():
    """Enhanced results tab with photo integration and dark mode support"""
    st.markdown("### 📋 Results & History")
    
    # Generated content
    if safe_get_session_state('content_generated', False):
        st.markdown("#### 📝 Generated Content & Photos")
        content = safe_get_session_state('generated_content', '')
        topic_photos = safe_get_session_state('topic_photos', {})
        
        # Action buttons
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.download_button(
                label="📥 Download Content",
                data=content,
                file_name=f"social_media_content_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
        with col2:
            if topic_photos:
                comprehensive_output = create_comprehensive_output_with_photos(
                    content, topic_photos, safe_get_session_state('sheets_data', '')
                )
                st.download_button(
                    label="📥 Download with Photos",
                    data=comprehensive_output,
                    file_name=f"content_with_photos_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
        with col3:
            if st.button("🔄 Regenerate Content"):
                safe_set_session_state('content_generated', False)
                safe_set_session_state('generated_content', '')
                safe_set_session_state('topic_photos', {})
                st.rerun()
        with col4:
            if st.button("🗑️ Clear All"):
                safe_set_session_state('content_generated', False)
                safe_set_session_state('generated_content', '')
                safe_set_session_state('posting_results', [])
                safe_set_session_state('topic_photos', {})
                st.rerun()
        
        # Content display with improved UI including photos
        render_generated_content_preview_with_photos(content, "results")
    else:
        st.info("ℹ️ No content generated yet. Go to 'Generate Content' tab to create content with photos.")
    
    # Posting results with photo information
    posting_results = safe_get_session_state('posting_results', [])
    if posting_results:
        st.markdown("#### 📱 Posting Results")
        
        # Summary metrics
        successful_posts = len([r for r in posting_results if r['result'].get('success')])
        total_posts = len(posting_results)
        posts_with_photos = len([r for r in posting_results if r.get('media_url')])
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Posts", total_posts)
        with col2:
            st.metric("Successful", successful_posts)
        with col3:
            st.metric("With Photos", posts_with_photos)
        with col4:
            st.metric("Failed", total_posts - successful_posts)
        
        # Detailed results with photo information
        for i, result in enumerate(posting_results, 1):
            platform = result['platform']
            topic = result['topic']
            post_result = result['result']
            content_preview = result.get('content', '')
            media_url = result.get('media_url')
            photo_info = result.get('photo_info')
            
            with st.expander(f"📱 {platform} - {topic} (Post {i}){'📸' if media_url else '📝'}"):
                if post_result.get('success'):
                    st.success("✅ Success!")
                    st.write(f"**Status:** {post_result.get('status', 'posted')}")
                    st.write(f"**Message:** {post_result.get('message', 'Successfully posted')}")
                    st.write(f"**Post ID:** {post_result.get('id', 'N/A')}")
                    if post_result.get('scheduled_at'):
                        st.write(f"**Scheduled for:** {post_result.get('scheduled_at')}")
                    
                    # Show photo information
                    if media_url:
                        st.write("**📸 Attached Photo:**")
                        col_a, col_b = st.columns([1, 2])
                        with col_a:
                            if photo_info:
                                st.image(photo_info['download_url'], width=150)
                        with col_b:
                            if photo_info:
                                st.write(f"**Description:** {photo_info['description']}")
                                st.write(f"**Photographer:** {photo_info['photographer']}")
                                st.write(f"**Platform Rec:** {photo_info['platform_recommendation']}")
                            st.code(media_url, language=None)
                else:
                    st.error("❌ Failed")
                    st.write(f"**Error:** {post_result.get('error', 'Unknown error')}")
                
                st.write("**Content Preview:**")
                st.text_area("", content_preview, height=100, disabled=True, key=f"preview_{i}")
    
    # Data preview
    dataframe = safe_get_session_state('dataframe', None)
    if dataframe is not None:
        st.markdown("#### 📊 Source Data")
        
        # Data summary
        st.write(f"**Rows:** {len(dataframe)} | **Columns:** {len(dataframe.columns)}")
        
        # Show dataframe
        st.dataframe(dataframe, use_container_width=True)
        
        # Export options
        col1, col2 = st.columns(2)
        with col1:
            csv_data = dataframe.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name=f"source_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        with col2:
            json_data = dataframe.to_json(orient='records', indent=2)
            st.download_button(
                label="📥 Download JSON",
                data=json_data,
                file_name=f"source_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

def render_analytics_tab():
    """Enhanced analytics tab with photo metrics and dark mode support"""
    st.markdown("### 📊 Analytics & Insights")
    
    # Content generation analytics
    content_count = safe_get_session_state('content_generated_count', 0)
    posting_results = safe_get_session_state('posting_results', [])
    topic_photos = safe_get_session_state('topic_photos', {})
    
    if content_count == 0 and not posting_results and not topic_photos:
        st.info("📈 Analytics will appear here after you generate and post content.")
        return
    
    # Content generation metrics
    st.markdown("#### 📝 Content Generation")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Sessions", content_count)
    with col2:
        total_posts = len(posting_results)
        st.metric("Total Posts", total_posts)
    with col3:
        successful_posts = len([r for r in posting_results if r['result'].get('success')])
        success_rate = (successful_posts / total_posts * 100) if total_posts > 0 else 0
        st.metric("Success Rate", f"{success_rate:.1f}%")
    with col4:
        platforms = set(r['platform'] for r in posting_results)
        st.metric("Platforms Used", len(platforms))
    with col5:
        total_photos = sum(len(photos) for photos in topic_photos.values())
        st.metric("Photos Found", total_photos)
    
    # Photo analytics
    if topic_photos:
        st.markdown("#### 📸 Photo Analytics")
        
        photo_data = []
        for topic, photos in topic_photos.items():
            instagram_photos = len([p for p in photos if 'Instagram' in p.get('platform_recommendation', '')])
            linkedin_photos = len([p for p in photos if 'LinkedIn' in p.get('platform_recommendation', '')])
            both_photos = len([p for p in photos if 'Both' in p.get('platform_recommendation', '')])
            
            photo_data.append({
                'Topic': topic,
                'Total Photos': len(photos),
                'Instagram Recommended': instagram_photos,
                'LinkedIn Recommended': linkedin_photos,
                'Both Platforms': both_photos
            })
        
        if photo_data:
            photo_df = pd.DataFrame(photo_data)
            st.dataframe(photo_df, use_container_width=True)
    
    # Platform breakdown with photo information
    if posting_results:
        st.markdown("#### 📱 Platform Breakdown")
        
        platform_data = {}
        for result in posting_results:
            platform = result['platform']
            success = result['result'].get('success', False)
            has_photo = bool(result.get('media_url'))
            
            if platform not in platform_data:
                platform_data[platform] = {'total': 0, 'successful': 0, 'with_photos': 0}
            
            platform_data[platform]['total'] += 1
            if success:
                platform_data[platform]['successful'] += 1
            if has_photo:
                platform_data[platform]['with_photos'] += 1
        
        for platform, data in platform_data.items():
            success_rate = (data['successful'] / data['total'] * 100) if data['total'] > 0 else 0
            photo_rate = (data['with_photos'] / data['total'] * 100) if data['total'] > 0 else 0
            
            st.write(f"**{platform}:**")
            st.write(f"  • Posts: {data['successful']}/{data['total']} successful ({success_rate:.1f}%)")
            st.write(f"  • Photos: {data['with_photos']}/{data['total']} with images ({photo_rate:.1f}%)")
    
    # Recent activity with photo information
    if posting_results:
        st.markdown("#### 🕒 Recent Activity")
        
        # Show last 5 posting attempts
        recent_results = posting_results[-5:] if len(posting_results) > 5 else posting_results
        
        for result in reversed(recent_results):
            platform = result['platform']
            topic = result['topic']
            success = result['result'].get('success', False)
            has_photo = bool(result.get('media_url'))
            
            status_icon = "✅" if success else "❌"
            photo_icon = "📸" if has_photo else "📝"
            timestamp = datetime.now().strftime("%H:%M:%S")  # In real app, you'd store actual timestamps
            
            st.write(f"{status_icon} {photo_icon} **{timestamp}** - {platform}: {topic}")

def render_photo_tab():
    """Photo management tab with dark mode support"""
    st.markdown("### 📸 Photo Gallery & Management")
    
    topic_photos = safe_get_session_state('topic_photos', {})
    sheets_data = safe_get_session_state('sheets_data', '')
    
    if not topic_photos and not sheets_data:
        st.info("📷 No photos available. Generate content first to see photo recommendations.")
        return
    
    # Manual photo search
    st.markdown("#### 🔍 Manual Photo Search")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        search_query = st.text_input("Search for photos:", placeholder="Enter keywords...")
    with col2:
        if st.button("🔍 Search Photos", type="primary"):
            if search_query:
                with st.spinner("Searching for photos..."):
                    manual_photos = search_stock_photos(search_query, 6)
                    safe_set_session_state('manual_search_photos', manual_photos)
                    safe_set_session_state('manual_search_query', search_query)
    
    # Show manual search results
    manual_photos = safe_get_session_state('manual_search_photos', [])
    manual_query = safe_get_session_state('manual_search_query', '')
    
    if manual_photos:
        st.markdown(f"#### 📸 Search Results for '{manual_query}'")
        render_photo_gallery(manual_photos, manual_query)
    
    # Show all topic photos
    if topic_photos:
        st.markdown("#### 🎯 Photos by Topic")
        
        # Topic selector
        selected_topic = st.selectbox(
            "Select a topic to view photos:",
            options=list(topic_photos.keys()),
            key="topic_photo_selector"
        )
        
        if selected_topic and topic_photos.get(selected_topic):
            photos = topic_photos[selected_topic]
            
            # Photo statistics for this topic
            instagram_photos = [p for p in photos if 'Instagram' in p.get('platform_recommendation', '')]
            linkedin_photos = [p for p in photos if 'LinkedIn' in p.get('platform_recommendation', '')]
            both_photos = [p for p in photos if 'Both' in p.get('platform_recommendation', '')]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Photos", len(photos))
            with col2:
                st.metric("Instagram", len(instagram_photos))
            with col3:
                st.metric("LinkedIn", len(linkedin_photos))
            with col4:
                st.metric("Both Platforms", len(both_photos))
            
            # Display photos
            render_photo_gallery(photos, selected_topic)
        
        # Photo summary
        st.markdown("#### 📊 Photo Summary")
        
        summary_data = []
        for topic, photos in topic_photos.items():
            instagram_count = len([p for p in photos if 'Instagram' in p.get('platform_recommendation', '')])
            linkedin_count = len([p for p in photos if 'LinkedIn' in p.get('platform_recommendation', '')])
            both_count = len([p for p in photos if 'Both' in p.get('platform_recommendation', '')])
            
            summary_data.append({
                'Topic': topic,
                'Total Photos': len(photos),
                'Instagram': instagram_count,
                'LinkedIn': linkedin_count,
                'Both Platforms': both_count
            })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main application function with enhanced photo integration and dark mode support"""
    try:
        # Render header
        render_header()
        
        # Render status metrics
        render_status_metrics()
        
        # Initialize LLM
        llm = initialize_llm()
        if not llm:
            st.error("❌ Failed to initialize LLM. Please check your Gemini API key.")
            st.markdown("""
            <div class="notification-box notification-error">
                <div class="notification-title">🔧 Setup Required</div>
                <div class="notification-content">Add your GOOGLE_API_KEY to the .env file to use the content generation features.</div>
            </div>
            """, unsafe_allow_html=True)
            return
        
        # Main tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Data Source", 
            "🎯 Generate Content", 
            "📱 Post Content", 
            "📸 Photo Gallery",
            "📋 Results",
            "📈 Analytics"
        ])
        
        with tab1:
            render_data_source_tab()
        
        with tab2:
            render_content_generation_tab()
        
        with tab3:
            render_posting_tab()
        
        with tab4:
            render_photo_tab()
        
        with tab5:
            render_results_tab()
        
        with tab6:
            render_analytics_tab()
            
    except Exception as e:
        st.error(f"❌ Application error: {str(e)}")
        logger.error(f"Main app error: {e}")
        
        # Show detailed error information for debugging
        with st.expander("🔧 Error Details (for debugging)"):
            st.write(f"**Error Type:** {type(e).__name__}")
            st.write(f"**Error Message:** {str(e)}")
            
            # Check common issues
            st.write("**Troubleshooting:**")
            api_status = validate_api_keys()
            
            if not api_status["gemini"]:
                st.write("- ❌ GOOGLE_API_KEY not found in environment")
            else:
                st.write("- ✅ GOOGLE_API_KEY found")
            
            if not api_status["buffer"]:
                st.write("- ⚠️ BUFFER_ACCESS_TOKEN not found (demo mode active)")
            else:
                st.write("- ✅ BUFFER_ACCESS_TOKEN found")
            
            if not api_status["unsplash"]:
                st.write("- ⚠️ UNSPLASH_ACCESS_KEY not found (demo mode for photos)")
            else:
                st.write("- ✅ UNSPLASH_ACCESS_KEY found")
            
            try:
                import crewai
                st.write("- ✅ CrewAI library installed")
            except ImportError:
                st.write("- ❌ CrewAI library not installed (run: pip install crewai)")

# ============================================================================
# APP ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Run main app directly (setup instructions removed)
    main()
