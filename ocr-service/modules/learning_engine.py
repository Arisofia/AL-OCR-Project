import logging
from typing import Dict, Any, Optional
from supabase import create_client, Client
from config import get_settings

logger = logging.getLogger("ocr-service.learning")

class LearningEngine:
    """
    Implements continuous learning by storing and retrieving document patterns,
    font characteristics, and correction history in Supabase.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.client: Optional[Client] = None
        
        if self.settings.supabase_url and self.settings.supabase_service_role:
            try:
                self.client = create_client(
                    self.settings.supabase_url, 
                    self.settings.supabase_service_role
                )
            except Exception as e:
                logger.error(f"Failed to initialize Supabase client: {e}")

    async def learn_from_result(self, doc_type: str, font_meta: Dict[str, Any], accuracy_score: float):
        """
        Records document characteristics for future reference and pattern matching.
        """
        if not self.client:
            return
            
        data = {
            "doc_type": doc_type,
            "font_metadata": font_meta,
            "accuracy_score": accuracy_score,
            "version": self.settings.version
        }
        
        try:
            # Upsert into a 'learning_patterns' table
            self.client.table("learning_patterns").upsert(data).execute()
        except Exception as e:
            logger.warning(f"Failed to record learning data: {e}")

    async def get_pattern_knowledge(self, doc_type: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves previously learned patterns for a specific document type.
        """
        if not self.client:
            return None
            
        try:
            result = self.client.table("learning_patterns") \
                .select("*") \
                .eq("doc_type", doc_type) \
                .order("accuracy_score", desc=True) \
                .limit(1) \
                .execute()
            
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Failed to fetch pattern knowledge: {e}")
            return None
