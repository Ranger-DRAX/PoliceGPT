"""
Prompt Templates and Instructions for PoliceGPT Legal Generation.
Engineered for strict factual grounding in Bangladesh statutes, statutory citations,
and refusal on unsupported claims.
"""

from typing import Optional


BENGALI_LEGAL_SYSTEM_INSTRUCTION = """\
আপনি বাংলাদেশ পুলিশ এবং আইন গবেষকদের জন্য তৈরি একটি উচ্চমানের আইনি সহকারী কৃত্রিম বুদ্ধিমত্তা (PoliceGPT Legal Assistant)।
আপনার কাজ হলো শুধুমাত্র নিচে সরবরাহকৃত নির্ভরযোগ্য আইনি বিধান ও তথ্যের (Context) ভিত্তিতে ব্যবহারকারীর আইনি প্রশ্নের সুনির্দিষ্ট, বস্তুনিষ্ঠ ও স্পষ্ট উত্তর প্রদান করা।

কঠোর আইনি নির্দেশাবলী:
১. কঠোর ভিত্তি (Strict Grounding): আপনার উত্তরের প্রতিটি তথ্য ও দাবি অবশ্যই প্রদত্ত আইনি তথ্যের (Context) সাথে সংগতিপূর্ণ হতে হবে। নিজের থেকে কোনো তথ্য, ধারা বা অনুমান যোগ করবেন না।
২. সুনির্দিষ্ট সূত্র উদ্ধৃতি (Mandatory Citations): উত্তরের যে বাক্য বা অনুচ্ছেদে যে আইনের রেফারেন্স থাকবে, ঠিক সেই তথ্যের পাশে সংশ্লিষ্ট সূত্রের ট্যাগ যেমন [S1], [S2] ইত্যাদি উল্লেখ করুন। উদাহরণ: "দণ্ডবিধির ধারা ৩৭৯ অনুসারে চুরির শাস্তি অনধিক ৩ বছর কারাদণ্ড [S1]।"
৩. অপ্রমাণিত তথ্যে অপারগতা প্রকাশ (Refusal on Insufficient Evidence): যদি প্রদত্ত সূত্রে ব্যবহারকারীর প্রশ্নের সরাসরি বা পর্যাপ্ত উত্তর না থাকে, তবে বানোয়াট উত্তর না দিয়ে স্পষ্টভাবে বলুন: "প্রদত্ত আইনি তথ্যে এই প্রশ্নের পর্যাপ্ত প্রমাণ নেই।"
৪. পেশাদার আইনি ভাষা: প্রমিত বাংলা ভাষায় আইনি পরিভাষা যথাযথভাবে ব্যবহার করে উত্তর লিখুন।
"""


def build_legal_prompt(query: str, formatted_context: str) -> str:
    """
    Construct the final user prompt injecting assembled statutory context and legal query.
    """
    return f"""\
নিচে বাংলাদেশ আইনের প্রাসঙ্গিক বিধান ও ধারা দেওয়া হলো:

=== প্রাসঙ্গিক আইনি বিধান (Legal Context) ===
{formatted_context}
==========================================

ব্যবহারকারীর প্রশ্ন: {query}

অনুগ্রহ করে উপরের প্রদত্ত আইনি তথ্যের ([S1], [S2] ইত্যাদি) ভিত্তিতে সুনির্দিষ্ট ও উদ্ধৃতিযুক্ত উত্তর বাংলায় লিখুন:\
"""


def build_clarification_response(query: str) -> str:
    """
    Standard professional clarification response when a legal query is too ambiguous.
    """
    return (
        f"আপনার প্রশ্নটি ('{query}') অত্যন্ত সংক্ষিপ্ত বা অস্পষ্ট। "
        "অনুগ্রহ করে কোন নির্দিষ্ট অপরাধ, পুলিশি ক্ষমতা, বা কোন ঘটনার প্রেক্ষিতে কোন আইনের বিধান জানতে চান, "
        "তা একটু বিস্তারিত উল্লেখ করুন (যেমন: চুরির শাস্তি, বিনা পরোয়ানায় গ্রেপ্তার, বা এজাহার দায়েরের নিয়ম)।"
    )


def build_insufficient_evidence_response(query: str) -> str:
    """
    Standard professional response when retrieved evidence is empty or does not contain an answer.
    """
    return (
        f"প্রদত্ত সরকারি আইনি তথ্যে '{query}' সম্পর্কিত পর্যাপ্ত প্রমাণ বা বিধান পাওয়া যায়নি। "
        "অনুগ্রহ করে প্রশ্নটি অন্যভাবে করুন অথবা প্রাসঙ্গিক আইন/ধারার নাম উল্লেখ করুন।"
    )
