from .base import BaseRetriever, register_retriever
import arxiv
from arxiv import Result as ArxivResult
from ..protocol import Paper
from ..utils import extract_markdown_from_pdf, extract_tex_code_from_tar
from tempfile import TemporaryDirectory
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import feedparser
from urllib.request import urlretrieve
from urllib.error import HTTPError, URLError
from tqdm import tqdm
import os
from loguru import logger

PDF_EXTRACT_TIMEOUT = 180


@register_retriever("arxiv")
class ArxivRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        if self.config.source.arxiv.category is None:
            raise ValueError("category must be specified for arxiv.")

    def _retrieve_raw_papers(self) -> list[ArxivResult]:
        client = arxiv.Client(num_retries=10, delay_seconds=10)
    
        categories = list(self.config.source.arxiv.category)
        include_cross_list = self.config.source.arxiv.get("include_cross_list", False)
    
        allowed_announce_types = {"new", "cross"} if include_cross_list else {"new"}
    
        all_ids = []
        seen_ids = set()
    
        for cat in categories:
            feed_url = f"https://rss.arxiv.org/atom/{cat}"
            feed = feedparser.parse(feed_url)
    
            logger.info(f"Fetching arXiv RSS for category: {cat}")
            logger.info(f"Feed title: {getattr(feed.feed, 'title', 'N/A')}")
            logger.info(f"RSS entries: {len(feed.entries)}")
    
            cat_ids = [
                entry.id.removeprefix("oai:arXiv.org:")
                for entry in feed.entries
                if entry.get("arxiv_announce_type", "new") in allowed_announce_types
            ]
    
            logger.info(f"Kept entries after announce_type filter for {cat}: {len(cat_ids)}")
    
            for pid in cat_ids:
                if pid not in seen_ids:
                    seen_ids.add(pid)
                    all_ids.append(pid)
    
        logger.info(f"Total unique arXiv ids collected from all categories: {len(all_ids)}")
    
        if self.config.executor.debug:
            all_ids = all_ids[:10]
    
        raw_papers = []
        bar = tqdm(total=len(all_ids))
        for i in range(0, len(all_ids), 20):
            search = arxiv.Search(id_list=all_ids[i:i + 20])
            batch = list(client.results(search))
            bar.update(len(batch))
            raw_papers.extend(batch)
        bar.close()
    
        return raw_papers

    def convert_to_paper(self, raw_paper: ArxivResult) -> Paper | None:
        title = raw_paper.title
        authors = [a.name for a in raw_paper.authors]
        abstract = raw_paper.summary
        pdf_url = raw_paper.pdf_url

        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                full_text = pool.submit(extract_text_from_pdf, raw_paper).result(
                    timeout=PDF_EXTRACT_TIMEOUT
                )
        except TimeoutError:
            logger.warning(f"PDF extraction timed out for {raw_paper.title}")
            full_text = None
        except Exception as e:
            logger.warning(f"PDF extraction failed for {raw_paper.title}: {e}")
            full_text = None

        if full_text is None:
            try:
                full_text = extract_text_from_tar(raw_paper)
            except Exception as e:
                logger.warning(f"TAR extraction failed for {raw_paper.title}: {e}")
                full_text = None

        return Paper(
            source=self.name,
            title=title,
            authors=authors,
            abstract=abstract,
            url=raw_paper.entry_id,
            pdf_url=pdf_url,
            full_text=full_text,
        )


def extract_text_from_pdf(paper: ArxivResult) -> str | None:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.pdf")
        if paper.pdf_url is None:
            logger.warning(f"No PDF URL available for {paper.title}")
            return None

        try:
            urlretrieve(paper.pdf_url, path)
        except (HTTPError, URLError, OSError) as e:
            logger.warning(f"Failed to download PDF for {paper.title}: {e}")
            return None

        try:
            full_text = extract_markdown_from_pdf(path)
        except Exception as e:
            logger.warning(f"Failed to extract full text of {paper.title} from pdf: {e}")
            full_text = None

        return full_text


def extract_text_from_tar(paper: ArxivResult) -> str | None:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.tar.gz")
        source_url = paper.source_url()
        if source_url is None:
            logger.warning(f"No source URL available for {paper.title}")
            return None

        try:
            urlretrieve(source_url, path)
        except (HTTPError, URLError, OSError) as e:
            logger.warning(f"Failed to download source tar for {paper.title}: {e}")
            return None

        try:
            file_contents = extract_tex_code_from_tar(path, paper.entry_id)
            if not file_contents or "all" not in file_contents:
                logger.warning(
                    f"Failed to extract full text of {paper.title} from tar: Main tex file not found."
                )
                return None
            full_text = file_contents["all"]
        except Exception as e:
            logger.warning(f"Failed to extract full text of {paper.title} from tar: {e}")
            full_text = None

        return full_text
