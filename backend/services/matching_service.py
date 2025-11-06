from typing import List, Dict
from backend.models.job_model import Job, MatchedJob
from backend.services.ml_service import MLService
import re
import logging

logger = logging.getLogger(__name__)


class MatchingService:
    def __init__(self, ml_service: MLService):
        self.ml_service = ml_service

    def create_user_profile(
        self, skills: List[str], keywords: str, experience: int
    ) -> str:
        """
        Create comprehensive user profile text for embedding
        Args:
            skills: list of user skills
            keywords: job keywords user is interested in
            experience: experience level ("entry", "mid", "senior")
        Returns:
            concatenated string
        """
        profile_parts = []

        if skills:
            profile_parts.append(f"Skills: {', '.join(skills)}")

        if keywords:
            profile_parts.append(f"Looking for: {keywords}")

        # --- normalize experience to a numeric "years" value ---
        years = 0

        # If we got a number already
        if isinstance(experience, (int, float)):
            years = int(experience)

        # If we got a string
        elif isinstance(experience, str):
            exp_str = experience.strip().lower()

            if exp_str.isdigit():
                years = int(exp_str)
            elif exp_str in ("entry", "junior"):
                years = 1
            elif exp_str in ("mid", "mid-level", "mid level"):
                years = 4
            elif exp_str in ("senior", "lead", "principal"):
                years = 7

        if years < 3:
            profile_parts.append("Entry level position, 0-2 years experience")
        elif years < 6:
            profile_parts.append("Mid-level position, 3-5 years experience")
        else:
            profile_parts.append("Senior position, 5+ years experience")

        return " ".join([p for p in profile_parts if p])

    def create_job_profile(self, job: Job) -> str:
        """
        Create comprehensive job description for embedding
        Args:
            job: Job object
        Returns:
            concatenated string
        """
        desc = job.description[:500] if len(job.description) > 500 else job.description
        return f"{job.title} at {job.company}. {desc}"

    def extract_matching_skills(
        self, user_skills: List[str], job_description: str
    ) -> List[str]:
        """
        Find which user skills appear in job description
        Args:
            user_skills: list of user skills
            job_description: full job description text
        Returns:
            list of matching skills
        """
        job_lower = job_description.lower()
        matching = []

        for skill in user_skills:
            skill_lower = skill.lower()
            # For skills with special chars (C++, C#, .NET), use simpler match
            if re.search(r"[^a-zA-Z0-9\s]", skill):
                # Direct substring search for special-char skills
                if skill_lower in job_lower:
                    matching.append(skill)
            else:
                # Use word boundaries for alphanumeric skills to avoid partial matches
                pattern = r"\b" + re.escape(skill_lower) + r"\b"
                if re.search(pattern, job_lower):
                    matching.append(skill)

        return matching

    def rank_jobs(
        self, user_data: Dict, jobs: List[Job], top_k: int = 20
    ) -> List[MatchedJob]:
        """
        Main ranking function using semantic similarity and skills matching
        Args:
            user_data: dict with keys "skills", "keywords", "experience"
            jobs: list of Job objects to rank
            top_k: number of top jobs to return
        Returns:
            list of MatchedJob objects with similarity scores and matching skills,
            sorted by final score
        """
        if not jobs:
            return []

        # Create user profile embedding
        user_profile = self.create_user_profile(
            user_data.get("skills", []),
            user_data.get("keywords", ""),
            user_data.get("experience", 0),
        )
        user_embedding = self.ml_service.encode_text(user_profile)

        # Create job embeddings
        job_profiles = [self.create_job_profile(job) for job in jobs]
        job_embeddings = [
            self.ml_service.encode_text(profile) for profile in job_profiles
        ]

        # Calculate similarities
        similarities = self.ml_service.batch_similarity(user_embedding, job_embeddings)

        # Create matched jobs with scores
        matched_jobs = []
        for job, similarity in zip(jobs, similarities):
            matching_skills = self.extract_matching_skills(
                user_data.get("skills", []), job.description
            )

            skills_match_ratio = (
                (len(matching_skills) / len(user_data.get("skills", [])))
                if user_data.get("skills", [])
                else 0
            )

            final_score = (0.8 * similarity) + (0.2 * skills_match_ratio)

            matched_jobs.append(
                MatchedJob(
                    job=job,
                    similarity_score=similarity,
                    matching_skills=matching_skills,
                    final_score=final_score,
                )
            )

        # Sort by final_score and return top K
        matched_jobs.sort(key=lambda x: x.final_score, reverse=True)
        return matched_jobs[:top_k]
