"""
Bilingual legal prompt templates enforcing strict context grounding and mandatory citations.
"""

from typing import List, Dict, Any


class LegalPromptBuilder:
    def __init__(self, system_version: str = "v1.2"):
        self.system_version = system_version

    def build_system_prompt(self, language: str = "bn") -> str:
        if language == "en":
            return (
                "You are PoliceGPT, an expert legal AI assistant specializing in the laws, penal codes, "
                "criminal procedures (CrPC), and Police Regulations of Bengal (PRB 1943) of Bangladesh.\n\n"
                "STRICT RULES:\n"
                "1. Answer ONLY based on the provided retrieved context. Do NOT fabricate or extrapolate laws.\n"
                "2. MANDATORY CITATIONS: Every legal assertion must explicitly cite the Act Name, Year, and Section/Rule/Dhara.\n"
                "3. If the context does not contain sufficient information to answer the question, state: "
                "'The provided legal records do not contain sufficient information to answer this query.'\n"
                "4. Maintain a neutral, professional, and objective legal tone.\n"
                "5. Include a standard legal disclaimer at the end."
            )

        # Bengali Default System Prompt
        return (
            "আপনি PoliceGPT, বাংলাদেশ পুলিশ রেগুলেশন (PRB ১৯৪৩), দণ্ডবিধি (Penal Code), ফৌজদারি কার্যবিধি (CrPC), "
            "এবং সংশ্লিষ্ট বিশেষ আইনসমূহের একজন বিশেষজ্ঞ আইনি সহকারী।\n\n"
            "কঠোর নির্দেশনাবলী:\n"
            "১. শুধুমাত্র নিচে দেওয়া প্রাসঙ্গিক আইনি অনুচ্ছেদ (Context)-এর তথ্যের ভিত্তিতে উত্তর প্রদান করুন। মনগড়া তথ্য দেবেন না।\n"
            "২. বাধ্যতামূলক সাইটেশন (Citations): প্রতিটি আইনি বক্তব্যের সাথে সুনির্দিষ্ট ধারা, বিধি, আইনের নাম ও বছর উল্লেখ করুন।\n"
            "৩. যদি প্রদত্ত তথ্যে উত্তর খুঁজে না পাওয়া যায়, তবে স্পষ্টভাবে বলুন: "
            "'প্রদত্ত আইনি নথিপত্রে এই বিষয়ে পর্যাপ্ত তথ্য পাওয়া যায়নি।'\n"
            "৪. আইনি ভাষা ও শালীনতা বজায় রেখে স্পষ্ট ও নির্ভুল পরামর্শ বা ব্যাখ্যা দিন।\n"
            "৫. উত্তরের শেষে আইনি ডিসক্লেইমার যুক্ত করুন।"
        )

    def format_context(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        formatted_passages = []
        for i, chunk in enumerate(retrieved_chunks):
            act_bn = chunk.get("act_name_bn") or chunk.get("act_name_en") or "বাংলাদেশ আইন"
            sec = chunk.get("section_number") or "N/A"
            sec_title = chunk.get("section_title") or ""
            content = chunk.get("content", "").strip()

            header = f"[উৎস {i+1} | আইন: {act_bn} | ধারা/বিধি: {sec} - {sec_title}]"
            formatted_passages.append(f"{header}\n{content}")

        return "\n\n".join(formatted_passages)

    def build_user_prompt(self, query: str, retrieved_chunks: List[Dict[str, Any]], language: str = "bn") -> str:
        context_str = self.format_context(retrieved_chunks)

        if language == "en":
            return (
                f"### RETRIEVED LEGAL CONTEXT:\n{context_str}\n\n"
                f"### CITIZEN / OFFICER QUERY:\n{query}\n\n"
                f"### INSTRUCTIONS:\n"
                f"Provide a comprehensive, accurate answer citing relevant Sections/Rules from the context above."
            )

        return (
            f"### প্রাসঙ্গিক আইনি তথ্য (Context):\n{context_str}\n\n"
            f"### প্রশ্ন/জিজ্ঞাসা:\n{query}\n\n"
            f"### উত্তর প্রদান নির্দেশনা:\n"
            f"উপরোক্ত প্রসঙ্গের আলোকে যথাযথ ধারা ও আইনের উল্লেখসহ বাংলায় পূর্ণাঙ্গ ও নির্ভরযোগ্য উত্তর দিন।"
        )
