"""Google Sheets integration for 3CB scoring system."""
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
from typing import List, Dict, Optional, Tuple
import os
import time
from urllib.parse import quote
from collections import deque
import functools


class RateLimiter:
    """
    Rate limiter for Google Sheets API.

    Google Sheets API limits:
    - 300 write requests per minute per user
    - 60 read requests per minute per user

    We'll be conservative and limit to 50 requests per minute.
    """

    def __init__(self, max_requests: int = 50, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window  # seconds
        self.requests = deque()

    def wait_if_needed(self):
        """Wait if we've exceeded the rate limit."""
        now = time.time()

        # Remove old requests outside the time window
        while self.requests and self.requests[0] < now - self.time_window:
            self.requests.popleft()

        # If we're at the limit, wait until the oldest request expires
        if len(self.requests) >= self.max_requests:
            sleep_time = self.requests[0] - (now - self.time_window) + 0.1
            if sleep_time > 0:
                time.sleep(sleep_time)
            # Clean up again after sleeping
            now = time.time()
            while self.requests and self.requests[0] < now - self.time_window:
                self.requests.popleft()

        # Record this request
        self.requests.append(time.time())

    def __call__(self, func):
        """Decorator to rate-limit a function."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self.wait_if_needed()
            return func(*args, **kwargs)
        return wrapper


def retry_on_quota_error(max_retries: int = 3, initial_delay: float = 5.0):
    """
    Decorator to retry on quota exceeded errors with exponential backoff.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except gspread.exceptions.APIError as e:
                    if '429' in str(e) or 'Quota exceeded' in str(e):
                        last_exception = e
                        if attempt < max_retries:
                            print(f"Rate limited, waiting {delay:.1f}s before retry {attempt + 1}/{max_retries}...")
                            time.sleep(delay)
                            delay *= 2  # Exponential backoff
                        continue
                    raise

            # If we've exhausted retries, raise the last exception
            if last_exception:
                raise last_exception
        return wrapper
    return decorator


# Global rate limiter instance
_rate_limiter = RateLimiter(max_requests=40, time_window=60)

# Path to history.csv (UTF-16 encoded)
HISTORY_CSV_PATH = os.path.join(os.path.dirname(__file__), 'data', 'history.csv')
# Path to swift.csv
SWIFT_CSV_PATH = os.path.join(os.path.dirname(__file__), 'data', 'swift.csv')


def normalize_deck(deck: str) -> str:
    """Normalize deck name by sorting cards alphabetically."""
    cards = [c.strip() for c in deck.split('|')]
    return ' | '.join(sorted(cards))


class HistoryLookup:
    """Lookup matchup results from history.csv."""

    _instance = None
    _cache = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._load_history()

    def _load_history(self):
        """Load and index history.csv."""
        self._cache = {}

        if not os.path.exists(HISTORY_CSV_PATH):
            return

        try:
            # Read UTF-16 encoded CSV
            df = pd.read_csv(HISTORY_CSV_PATH, encoding='utf-16', sep='\t')

            # Build lookup: (normalized_deck1, normalized_deck2, turn_order) -> result
            for _, row in df.iterrows():
                deck1 = normalize_deck(str(row.get('Decklist ', '') or row.get('Decklist', '')))
                deck2 = normalize_deck(str(row.get('Opponent Decklist', '')))
                turn_order = str(row.get('Turn Order', ''))
                result = str(row.get('Result', ''))

                if deck1 and deck2 and result:
                    # If 1st, deck1 is on play; if 2nd, deck1 is on draw
                    if turn_order == '1st':
                        key = (deck1, deck2)  # deck1 on play
                    else:
                        key = (deck2, deck1)  # deck1 on draw, so deck2 is on play

                    # Convert result to 0/1/2 from deck on play's perspective
                    if turn_order == '1st':
                        val = 2 if result == 'Win' else (0 if result == 'Loss' else 1)
                    else:
                        # deck1 was on draw, result is from their perspective
                        # so we invert for on-play perspective
                        val = 0 if result == 'Win' else (2 if result == 'Loss' else 1)

                    self._cache[key] = val

        except Exception as e:
            print(f"Error loading history.csv: {e}")

    def lookup(self, deck1_on_play: str, deck2_on_draw: str) -> Optional[int]:
        """
        Look up the accepted result for a matchup.

        Args:
            deck1_on_play: Deck that is on play
            deck2_on_draw: Deck that is on draw

        Returns:
            0=Loss, 1=Tie, 2=Win from deck1's perspective, or None if not found
        """
        key = (normalize_deck(deck1_on_play), normalize_deck(deck2_on_draw))
        return self._cache.get(key)

    def reload(self):
        """Reload history from disk."""
        self._load_history()


class SwiftLookup:
    """
    Lookup matchup results from swift.csv matrix.

    Swift scoring system:
    - 4 = Win on Play (2) + Win on Draw (2)
    - 3 = Win on Play (2) + Tie on Draw (1)
    - 1 = Tie on Play (1) + Loss on Draw (0)
    - 0 = Loss on Play (0) + Loss on Draw (0)
    - 2 = Ambiguous: could be Win+Loss (2+0) or Tie+Tie (1+1)

    For score=2, we can't determine individual play/draw results,
    so we return None for those but can validate the sum.
    """

    _instance = None
    _cache = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._load_swift()

    def _load_swift(self):
        """Load and index swift.csv matrix."""
        self._cache = {}  # (deck1, deck2) -> {'total': int, 'on_play': int or None, 'on_draw': int or None}

        if not os.path.exists(SWIFT_CSV_PATH):
            return

        try:
            # Read the CSV
            with open(SWIFT_CSV_PATH, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            if len(lines) < 6:
                return

            # Row 0 has deck names starting at column 8
            header_line = lines[0].strip()
            # Parse CSV properly handling quoted fields
            import csv
            reader = csv.reader([header_line])
            header_row = list(reader)[0]

            # Deck names start at column 8 (0-indexed)
            deck_columns = header_row[8:]

            # Clean up deck names (remove 'test:' prefix if present)
            deck_columns = [d.replace('test: ', '').strip() for d in deck_columns]

            # Data rows start at row 5 (0-indexed), after header rows
            data_start = 5

            for line in lines[data_start:]:
                if not line.strip():
                    continue

                reader = csv.reader([line])
                row = list(reader)[0]

                if len(row) < 9:
                    continue

                # Column 1 has the deck name for this row
                row_deck = row[1].strip()
                if not row_deck or row_deck.startswith('test:'):
                    # Skip test decks
                    if row_deck.startswith('test:'):
                        row_deck = row_deck.replace('test: ', '').strip()
                    else:
                        continue

                # Normalize the row deck name
                row_deck_norm = normalize_deck(row_deck)

                # Scores start at column 8
                for j, score_str in enumerate(row[8:]):
                    if j >= len(deck_columns):
                        break

                    col_deck = deck_columns[j]
                    if not col_deck:
                        continue

                    col_deck_norm = normalize_deck(col_deck)

                    # Skip mirrors
                    if row_deck_norm == col_deck_norm:
                        continue

                    score_str = score_str.strip()
                    if not score_str or score_str == '-1':
                        continue

                    try:
                        total = int(score_str)
                    except ValueError:
                        continue

                    if total < 0 or total > 4:
                        continue

                    # Determine individual play/draw scores
                    # Row deck is on play vs column deck
                    if total == 4:
                        on_play, on_draw = 2, 2  # W, W
                    elif total == 3:
                        on_play, on_draw = 2, 1  # W, T
                    elif total == 1:
                        on_play, on_draw = 1, 0  # T, L
                    elif total == 0:
                        on_play, on_draw = 0, 0  # L, L
                    elif total == 2:
                        # Ambiguous - default to T+T (user preference)
                        on_play, on_draw = 1, 1  # T, T
                    else:
                        continue

                    # Store: row_deck on play vs col_deck
                    key = (row_deck_norm, col_deck_norm)
                    self._cache[key] = {
                        'total': total,
                        'on_play': on_play,
                        'on_draw': on_draw
                    }

        except Exception as e:
            print(f"Error loading swift.csv: {e}")
            import traceback
            traceback.print_exc()

    def lookup(self, deck1_on_play: str, deck2_on_draw: str) -> Optional[Dict]:
        """
        Look up Swift's result for a matchup.

        Args:
            deck1_on_play: Deck that is on play
            deck2_on_draw: Deck that is on draw

        Returns:
            Dict with 'total', 'on_play', 'on_draw' or None if not found.
            For score=2 matchups, on_play and on_draw will be None.
        """
        key = (normalize_deck(deck1_on_play), normalize_deck(deck2_on_draw))
        return self._cache.get(key)

    def lookup_on_play(self, deck1_on_play: str, deck2_on_draw: str) -> Optional[int]:
        """
        Look up Swift's on-play result for a matchup.

        Returns:
            0=Loss, 1=Tie, 2=Win from deck1's perspective, or None if not found/ambiguous.
        """
        result = self.lookup(deck1_on_play, deck2_on_draw)
        if result is None:
            return None
        return result.get('on_play')

    def get_all_matchups(self) -> List[Tuple[str, str, Dict]]:
        """Get all matchups with their scores."""
        return [(k[0], k[1], v) for k, v in self._cache.items()]

    def reload(self):
        """Reload swift data from disk."""
        self._load_swift()


SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive'
]

# Sheet names
DECKS_SHEET = "Decks"
MATRIX_ONPLAY_SHEET = "Matrix_OnPlay"
MATRIX_ONDRAW_SHEET = "Matrix_OnDraw"
GAME_MATRIX_SHEET = "Game_Matrix"
NASH_SHEET = "Nash"
RESULTS_SHEET = "Results"

# Cache TTL in seconds (set very high - data loaded on connect, refreshed manually)
CACHE_TTL = 3600  # 1 hour - effectively infinite for a session


class SheetsManager:
    """Manages Google Sheets connection and operations for 3CB scoring."""

    def __init__(self, sheet_url: str, creds_path: str = "credentials.json"):
        """
        Initialize connection to Google Sheet.

        Args:
            sheet_url: URL of the Google Sheet
            creds_path: Path to service account credentials JSON
        """
        self.sheet_url = sheet_url
        self.creds_path = creds_path
        self.spreadsheet = None

        # Cache for data (loaded on connect, persists for session)
        self._decks_cache = None
        self._decks_cache_time = 0
        self._results_cache = None
        self._results_cache_time = 0
        self._results_raw_cache = None  # Raw sheet data for operations
        self._headers_cache = None
        self._headers_cache_time = 0
        self._goldfish_cache = None
        self._goldfish_cache_time = 0

        self._connect()

    def _connect(self):
        """Establish connection to Google Sheets and download data locally."""
        if not os.path.exists(self.creds_path):
            raise FileNotFoundError(
                f"Credentials file not found: {self.creds_path}\n"
                "Please download service account credentials from Google Cloud Console."
            )

        creds = Credentials.from_service_account_file(self.creds_path, scopes=SCOPES)
        gc = gspread.authorize(creds)
        self.spreadsheet = gc.open_by_url(self.sheet_url)

        # Pre-load all data locally on connect
        self._preload_data()

    def _preload_data(self):
        """Download all sheet data locally for fast access."""
        import time as t
        now = t.time()

        # Load Results sheet
        try:
            sheet = self.spreadsheet.worksheet(RESULTS_SHEET)
            _rate_limiter.wait_if_needed()
            data = sheet.get_all_values()
            if data:
                self._results_raw_cache = data  # Store raw data
                self._headers_cache = data[0] if data else []
                self._headers_cache_time = now
                # Build DataFrame
                import pandas as pd
                if len(data) > 1:
                    df = pd.DataFrame(data[1:], columns=data[0])
                    # Normalize deck names
                    if 'Deck1_OnPlay' in df.columns:
                        df['Deck1_OnPlay'] = df['Deck1_OnPlay'].apply(lambda x: normalize_deck(x) if x else '')
                    if 'Deck2_OnDraw' in df.columns:
                        df['Deck2_OnDraw'] = df['Deck2_OnDraw'].apply(lambda x: normalize_deck(x) if x else '')
                    self._results_cache = df
                    self._results_cache_time = now
        except gspread.WorksheetNotFound:
            pass

        # Load Decks sheet
        try:
            sheet = self.spreadsheet.worksheet(DECKS_SHEET)
            _rate_limiter.wait_if_needed()
            data = sheet.get_all_values()
            if data and len(data) > 1:
                decks = []
                for row in data[1:]:
                    if row and row[0].strip():
                        decks.append(normalize_deck(row[0].strip()))
                self._decks_cache = decks
                self._decks_cache_time = now
                # Also cache goldfish data
                self._goldfish_cache = {}
                for row in data[1:]:
                    if len(row) >= 2 and row[0].strip():
                        deck = normalize_deck(row[0].strip())
                        goldfish = row[1].strip() if len(row) > 1 else ''
                        if goldfish:
                            try:
                                self._goldfish_cache[deck] = int(goldfish)
                            except ValueError:
                                pass
                self._goldfish_cache_time = now
        except gspread.WorksheetNotFound:
            pass

    def _get_or_create_sheet(self, name: str, rows: int = 1000, cols: int = 26) -> gspread.Worksheet:
        """Get worksheet by name, creating it if it doesn't exist."""
        try:
            return self.spreadsheet.worksheet(name)
        except gspread.WorksheetNotFound:
            return self.spreadsheet.add_worksheet(title=name, rows=rows, cols=cols)

    @retry_on_quota_error(max_retries=5, initial_delay=10.0)
    def _rate_limited_batch_update(self, sheet: gspread.Worksheet, updates: List[Dict], value_input_option: str = 'USER_ENTERED'):
        """
        Perform a rate-limited batch update with retry on quota errors.

        Args:
            sheet: The worksheet to update
            updates: List of update dicts with 'range' and 'values'
            value_input_option: How to interpret input data
        """
        _rate_limiter.wait_if_needed()
        # Deep copy updates since gspread modifies the range dicts in place
        # (adds sheet name prefix), which corrupts them on retry
        import copy
        updates_copy = copy.deepcopy(updates)
        sheet.batch_update(updates_copy, value_input_option=value_input_option)

    def _chunked_batch_update(self, sheet: gspread.Worksheet, updates: List[Dict],
                               chunk_size: int = 100, value_input_option: str = 'USER_ENTERED',
                               progress_callback=None, progress_start: int = 50, progress_end: int = 90):
        """
        Perform batch updates in chunks with rate limiting.

        Args:
            sheet: The worksheet to update
            updates: List of update dicts with 'range' and 'values'
            chunk_size: Number of updates per batch (smaller = safer for rate limits)
            value_input_option: How to interpret input data
            progress_callback: Optional callback(current, total, status_text) for progress
            progress_start: Starting progress percentage
            progress_end: Ending progress percentage
        """
        total_chunks = (len(updates) + chunk_size - 1) // chunk_size
        for i, start in enumerate(range(0, len(updates), chunk_size)):
            chunk = updates[start:start + chunk_size]
            if progress_callback:
                pct = progress_start + int((i / total_chunks) * (progress_end - progress_start))
                progress_callback(pct, 100, f"Writing batch {i+1}/{total_chunks}...")
            self._rate_limited_batch_update(sheet, chunk, value_input_option)

    # =========================================================================
    # Decks Sheet Operations
    # =========================================================================

    def read_decks(self, force_refresh: bool = False) -> List[str]:
        """Read deck names from Decks sheet (cached), normalized alphabetically."""
        now = time.time()
        if not force_refresh and self._decks_cache is not None and (now - self._decks_cache_time) < CACHE_TTL:
            return self._decks_cache

        sheet = self._get_or_create_sheet(DECKS_SHEET)
        values = sheet.col_values(1)  # Column A
        # Filter out empty values and header if present, normalize deck names
        decks = []
        for v in values:
            v = v.strip()
            if v and v.lower() != 'deck':
                decks.append(normalize_deck(v))

        self._decks_cache = decks
        self._decks_cache_time = now
        return decks

    def invalidate_cache(self):
        """Invalidate all caches to force refresh on next read."""
        self._decks_cache = None
        self._results_cache = None
        self._results_raw_cache = None
        self._headers_cache = None
        self._goldfish_cache = None

    def refresh_data(self):
        """Re-download all data from sheets (call after external changes)."""
        self.invalidate_cache()
        self._preload_data()

    def get_goldfish(self, deck: str) -> Optional[str]:
        """Get goldfish turn for a deck (cached)."""
        now = time.time()
        if self._goldfish_cache is None or (now - self._goldfish_cache_time) >= CACHE_TTL:
            # Load goldfish data
            self._goldfish_cache = {}
            try:
                sheet = self._get_or_create_sheet(DECKS_SHEET)
                data = sheet.get_all_values()
                if len(data) > 1 and len(data[0]) >= 2:
                    for row in data[1:]:  # Skip header
                        if len(row) >= 2 and row[0].strip():
                            deck_name = row[0].strip()
                            goldfish = row[1].strip() if row[1] else None
                            # Normalize deck name for lookup
                            self._goldfish_cache[normalize_deck(deck_name)] = goldfish
                self._goldfish_cache_time = now
            except Exception:
                pass

        return self._goldfish_cache.get(normalize_deck(deck))

    def write_decks(self, decks: List[str]):
        """Write deck names to Decks sheet."""
        sheet = self._get_or_create_sheet(DECKS_SHEET)
        sheet.clear()
        sheet.update('A1', [['Deck']] + [[d] for d in decks])

    def _build_scryfall_url(self, deck: str) -> str:
        """Build a Scryfall search URL for a deck."""
        cards = [c.strip() for c in deck.split('|')]
        # Build query: (!"Card1" or !"Card2" or !"Card3")
        query_parts = [f'!"{card}"' for card in cards]
        query = '(' + ' or '.join(query_parts) + ')'
        return f'https://scryfall.com/search?q={quote(query)}'

    def update_deck_links(self):
        """Update the Decks sheet so each deck name is a hyperlink to Scryfall.

        Uses cached deck list - only makes API call to write updates.
        """
        # Use cached decks instead of API call
        decks = self.read_decks()
        if not decks:
            return

        sheet = self._get_or_create_sheet(DECKS_SHEET)

        # Build all hyperlink formulas locally
        updates = []
        for i, deck_name in enumerate(decks):
            url = self._build_scryfall_url(deck_name)
            formula = f'=HYPERLINK("{url}", "{deck_name}")'
            updates.append({
                'range': f'A{i + 2}',  # +2 for header and 1-indexing
                'values': [[formula]]
            })

        if updates:
            self._chunked_batch_update(sheet, updates)

    def normalize_all_deck_names(self):
        """
        Normalize all deck names in Decks and Results sheets to alphabetical order.
        Uses cached data for reads, only makes API calls to write updates.
        """
        # Normalize Decks sheet - use cached decks
        decks = self.read_decks()  # Already normalized
        # Decks are already normalized on read, so just update links
        # (done separately in update_deck_links)

        # Normalize Results sheet - use cached raw data if available
        if self._results_raw_cache:
            results_data = self._results_raw_cache
        else:
            results_sheet = self._get_or_create_sheet(RESULTS_SHEET)
            _rate_limiter.wait_if_needed()
            results_data = results_sheet.get_all_values()

        if results_data and len(results_data) > 1:
            results_updates = []
            for i, row in enumerate(results_data[1:], start=2):
                if len(row) >= 2:
                    d1_original = row[0].strip() if row[0] else ''
                    d2_original = row[1].strip() if row[1] else ''

                    d1_normalized = normalize_deck(d1_original) if d1_original else ''
                    d2_normalized = normalize_deck(d2_original) if d2_original else ''

                    if d1_original != d1_normalized:
                        results_updates.append({
                            'range': f'A{i}',
                            'values': [[d1_normalized]]
                        })
                    if d2_original != d2_normalized:
                        results_updates.append({
                            'range': f'B{i}',
                            'values': [[d2_normalized]]
                        })

            if results_updates:
                results_sheet = self._get_or_create_sheet(RESULTS_SHEET)
                self._chunked_batch_update(results_sheet, results_updates)
                self.invalidate_cache()

    # =========================================================================
    # Results Sheet Operations
    # =========================================================================

    def read_results(self, force_refresh: bool = False) -> pd.DataFrame:
        """Read results from Results sheet (cached), with normalized deck names."""
        now = time.time()
        if not force_refresh and self._results_cache is not None and (now - self._results_cache_time) < CACHE_TTL:
            return self._results_cache  # Return directly, no copy needed for read-only ops

        sheet = self._get_or_create_sheet(RESULTS_SHEET)
        _rate_limiter.wait_if_needed()
        data = sheet.get_all_values()

        if not data or len(data) < 2:
            # Return empty DataFrame with basic columns
            df = pd.DataFrame(columns=['Deck1_OnPlay', 'Deck2_OnDraw', 'Consensus', 'Discrepancy'])
        else:
            headers = data[0]
            rows = data[1:]
            df = pd.DataFrame(rows, columns=headers)
            # Normalize deck names
            if 'Deck1_OnPlay' in df.columns:
                df['Deck1_OnPlay'] = df['Deck1_OnPlay'].apply(lambda x: normalize_deck(x) if x else x)
            if 'Deck2_OnDraw' in df.columns:
                df['Deck2_OnDraw'] = df['Deck2_OnDraw'].apply(lambda x: normalize_deck(x) if x else x)

        self._results_cache = df
        self._results_cache_time = now
        self._headers_cache = list(df.columns) if not df.empty else None
        self._headers_cache_time = now
        return df

    def get_scorer_columns(self) -> List[str]:
        """Get list of scorer column names from Results sheet."""
        df = self.read_results()
        # Scorer columns are everything except the fixed columns
        fixed_cols = {'Deck1_OnPlay', 'Deck2_OnDraw', 'Consensus', 'Discrepancy', 'Accepted'}
        return [col for col in df.columns if col not in fixed_cols]

    def initialize_results_sheet(self, decks: List[str]):
        """Initialize Results sheet with all matchup pairs."""
        sheet = self._get_or_create_sheet(RESULTS_SHEET)

        # Generate all pairs
        pairs = []
        for d1 in decks:
            for d2 in decks:
                if d1 != d2:  # Skip mirror matches or include them?
                    pairs.append([d1, d2, '', ''])  # Deck1, Deck2, Consensus, Discrepancy

        # Write headers and data
        headers = ['Deck1_OnPlay', 'Deck2_OnDraw', 'Consensus', 'Discrepancy']
        sheet.clear()
        sheet.update('A1', [headers] + pairs)

    def sync_results_with_decks(self) -> int:
        """
        Add any missing matchups to Results sheet for new decks.

        Returns:
            Number of new matchups added.
        """
        decks = self.read_decks(force_refresh=True)
        df = self.read_results(force_refresh=True)

        if df.empty:
            # No results yet, initialize
            self.initialize_results_sheet(decks)
            return len(decks) * (len(decks) - 1)

        # Find existing matchups
        existing = set()
        for _, row in df.iterrows():
            d1 = row.get('Deck1_OnPlay', '')
            d2 = row.get('Deck2_OnDraw', '')
            if d1 and d2:
                existing.add((d1, d2))

        # Find missing matchups
        missing = []
        for d1 in decks:
            for d2 in decks:
                if d1 != d2 and (d1, d2) not in existing:
                    missing.append([d1, d2, '', ''])

        if not missing:
            return 0

        # Append missing matchups to sheet
        sheet = self._get_or_create_sheet(RESULTS_SHEET)

        # Find the next empty row
        next_row = len(df) + 2  # +1 for header, +1 for 1-indexing

        # Append the missing rows
        sheet.update(f'A{next_row}', missing)

        # Invalidate cache since we added data
        self.invalidate_cache()

        return len(missing)

    def update_accepted_column(self):
        """
        Update the Accepted column in Results sheet from history.csv.
        Uses cached data for reads, only makes API calls to write.
        """
        history = HistoryLookup.get_instance()
        history.reload()  # Reload to get latest history.csv

        sheet = self._get_or_create_sheet(RESULTS_SHEET)

        # Use cached headers
        headers = self._headers_cache if self._headers_cache else sheet.row_values(1)

        # Add Accepted column if it doesn't exist
        if 'Accepted' not in headers:
            # Insert before Consensus column
            try:
                consensus_idx = headers.index('Consensus')
            except ValueError:
                consensus_idx = len(headers)
            _rate_limiter.wait_if_needed()
            sheet.insert_cols([['Accepted'] + [''] * (sheet.row_count - 1)], consensus_idx + 1)
            headers = sheet.row_values(1)  # Refresh headers
            self._headers_cache = headers
            self._results_raw_cache = None  # Structure changed

        accepted_col_idx = headers.index('Accepted') + 1  # 1-indexed

        # Use cached raw data if available
        if self._results_raw_cache:
            data = self._results_raw_cache
        else:
            _rate_limiter.wait_if_needed()
            data = sheet.get_all_values()

        if len(data) < 2:
            return

        # Build updates for Accepted column
        updates = []
        for i, row in enumerate(data[1:], start=2):  # Start at row 2 (1-indexed, skip header)
            if len(row) < 2:
                continue
            deck1 = row[0].strip()
            deck2 = row[1].strip()

            if not deck1 or not deck2:
                continue

            accepted = history.lookup(deck1, deck2)
            if accepted is not None:
                updates.append({
                    'range': f'{gspread.utils.rowcol_to_a1(i, accepted_col_idx)}',
                    'values': [[str(accepted)]]
                })

        if updates:
            self._chunked_batch_update(sheet, updates)

        # Invalidate cache
        self.invalidate_cache()

    def import_swift_scores(self, progress_callback=None) -> Dict[str, int]:
        """
        Import Swift scores from swift.csv as a scorer column.

        Builds the entire Swift column locally and pushes it in one batch.

        Args:
            progress_callback: Optional callback(current, total, status_text) for progress updates

        Returns:
            Dict with counts: 'imported', 'skipped_ambiguous', 'skipped_missing', 'sum_mismatches'
        """
        swift = SwiftLookup.get_instance()
        swift.reload()

        if progress_callback:
            progress_callback(0, 100, "Loading sheet data...")

        sheet = self._get_or_create_sheet(RESULTS_SHEET)
        all_data = sheet.get_all_values()

        if len(all_data) < 2:
            return {'imported': 0, 'skipped_ambiguous': 0, 'skipped_missing': 0,
                    'skipped_existing': 0, 'sum_mismatches': 0}

        headers = all_data[0]

        # Ensure Swift column exists
        scorer_name = 'Swift'
        if scorer_name not in headers:
            self.add_scorer_column(scorer_name)
            all_data = sheet.get_all_values()
            headers = all_data[0]

        swift_col_idx = headers.index(scorer_name)  # 0-indexed for list access
        deck1_col = headers.index('Deck1_OnPlay')
        deck2_col = headers.index('Deck2_OnDraw')

        if progress_callback:
            progress_callback(10, 100, "Building Swift column...")

        counts = {'imported': 0, 'skipped_ambiguous': 0, 'skipped_missing': 0, 'skipped_existing': 0}

        # Build the entire Swift column
        swift_column = [['Swift']]  # Header
        total_rows = len(all_data) - 1

        for i, row in enumerate(all_data[1:]):
            if progress_callback and i % 500 == 0:
                pct = 10 + int((i / total_rows) * 40)
                progress_callback(pct, 100, f"Processing row {i}/{total_rows}...")

            # Pad row if needed
            while len(row) <= max(swift_col_idx, deck1_col, deck2_col):
                row.append('')

            deck1 = row[deck1_col].strip()
            deck2 = row[deck2_col].strip()
            existing_swift = row[swift_col_idx].strip()

            # Keep existing value if present
            if existing_swift != '':
                swift_column.append([existing_swift])
                counts['skipped_existing'] += 1
                continue

            if not deck1 or not deck2:
                swift_column.append([''])
                continue

            # Skip mirrors
            if normalize_deck(deck1) == normalize_deck(deck2):
                swift_column.append([''])
                continue

            # Look up Swift's score
            swift_result = swift.lookup(deck1, deck2)

            if swift_result is None:
                swift_column.append([''])
                counts['skipped_missing'] += 1
                continue

            # Skip ambiguous scores (total=2 could be W+L or T+T)
            if swift_result.get('total') == 2:
                swift_column.append([''])
                counts['skipped_ambiguous'] += 1
                continue

            on_play_score = swift_result.get('on_play')
            swift_column.append([str(on_play_score)])
            counts['imported'] += 1

        if progress_callback:
            progress_callback(55, 100, "Writing Swift column to sheet...")

        # Write the entire column in one call
        col_letter = gspread.utils.rowcol_to_a1(1, swift_col_idx + 1)[0]  # Get column letter
        # Handle multi-letter columns
        col_letter = gspread.utils.rowcol_to_a1(1, swift_col_idx + 1).rstrip('0123456789')
        range_str = f"{col_letter}1:{col_letter}{len(swift_column)}"

        _rate_limiter.wait_if_needed()
        sheet.update(range_str, swift_column, value_input_option='USER_ENTERED')

        self.invalidate_cache()

        if progress_callback:
            progress_callback(70, 100, "Updating consensus...")

        self._update_all_consensus_discrepancy()

        if progress_callback:
            progress_callback(100, 100, "Done!")

        return counts

    def validate_swift_ambiguous_sums(self) -> int:
        """
        Validate that ambiguous Swift matchups (total=2) have matching sums.

        For each matchup where Swift has total=2:
        - Get the on-play result from Results (Accepted or Consensus)
        - Get the on-draw result (from inverse matchup)
        - Check if on_play + on_draw == 2
        - If not, flag as discrepancy

        Returns:
            Number of sum mismatches found.
        """
        swift = SwiftLookup.get_instance()
        df = self.read_results(force_refresh=True)

        if df.empty:
            return 0

        sheet = self._get_or_create_sheet(RESULTS_SHEET)
        headers = sheet.row_values(1)

        # Find Discrepancy column
        try:
            discrepancy_col = headers.index('Discrepancy') + 1
        except ValueError:
            return 0

        # Build lookup for results: (deck1, deck2) -> result
        results_lookup = {}
        row_lookup = {}  # (deck1, deck2) -> row_idx
        for idx, row in df.iterrows():
            d1 = normalize_deck(row.get('Deck1_OnPlay', ''))
            d2 = normalize_deck(row.get('Deck2_OnDraw', ''))
            # Use Accepted if available, otherwise Consensus
            result = row.get('Accepted', '')
            if result == '':
                result = row.get('Consensus', '')
            if d1 and d2 and result != '':
                try:
                    results_lookup[(d1, d2)] = int(result)
                except ValueError:
                    pass
            if d1 and d2:
                row_lookup[(d1, d2)] = idx + 2  # +2 for 1-indexing and header

        mismatch_count = 0
        updates = []

        # Check all ambiguous Swift matchups
        for d1, d2, swift_data in swift.get_all_matchups():
            if swift_data['total'] != 2:
                continue  # Only check ambiguous ones

            d1_norm = normalize_deck(d1)
            d2_norm = normalize_deck(d2)

            # Get on-play result (d1 on play vs d2)
            on_play = results_lookup.get((d1_norm, d2_norm))

            # Get on-draw result (d1 on draw vs d2)
            # This is 2 - (d2's result when on play vs d1)
            d2_on_play_vs_d1 = results_lookup.get((d2_norm, d1_norm))
            on_draw = (2 - d2_on_play_vs_d1) if d2_on_play_vs_d1 is not None else None

            # Check if both results exist
            if on_play is not None and on_draw is not None:
                total = on_play + on_draw
                if total != 2:
                    # Sum mismatch - flag as discrepancy
                    mismatch_count += 1
                    row_idx = row_lookup.get((d1_norm, d2_norm))
                    if row_idx:
                        updates.append({
                            'range': gspread.utils.rowcol_to_a1(row_idx, discrepancy_col),
                            'values': [['TRUE']]
                        })

        if updates:
            self._chunked_batch_update(sheet, updates)
            self.invalidate_cache()

        return mismatch_count

    def _update_all_consensus_discrepancy(self):
        """Update Consensus and Discrepancy columns for all rows (bulk update).

        Uses cached data for reading, only makes API call to write.
        """
        from collections import Counter

        # Use cached raw data if available, otherwise fetch
        if self._results_raw_cache:
            data = [row[:] for row in self._results_raw_cache]  # Copy to avoid mutation
        else:
            sheet = self._get_or_create_sheet(RESULTS_SHEET)
            _rate_limiter.wait_if_needed()
            data = sheet.get_all_values()

        if len(data) < 2:
            return

        headers = data[0]

        # Find column indices (0-indexed)
        try:
            consensus_col_idx = headers.index('Consensus')
            discrepancy_col_idx = headers.index('Discrepancy')
            deck1_col = headers.index('Deck1_OnPlay')
            deck2_col = headers.index('Deck2_OnDraw')
        except ValueError:
            return

        # Find scorer columns (exclude fixed columns but INCLUDE Accepted for discrepancy calc)
        fixed_cols = {'Deck1_OnPlay', 'Deck2_OnDraw', 'Consensus', 'Discrepancy'}
        scorer_indices = [i for i, h in enumerate(headers) if h and h not in fixed_cols]
        scorer_names = [headers[i] for i in scorer_indices]

        # Build columns locally
        consensus_column = [['Consensus']]
        discrepancy_column = [['Discrepancy']]

        for row in data[1:]:
            while len(row) < len(headers):
                row.append('')

            d1 = normalize_deck(row[deck1_col].strip()) if row[deck1_col].strip() else ''
            d2 = normalize_deck(row[deck2_col].strip()) if row[deck2_col].strip() else ''

            # Get scores from all scorers for this row (excluding Accepted for consensus)
            scores_for_consensus = []
            all_scores = []  # Including Accepted, for discrepancy
            for idx, name in zip(scorer_indices, scorer_names):
                if idx < len(row) and row[idx].strip():
                    try:
                        val = int(row[idx])
                        all_scores.append(val)
                        if name != 'Accepted':
                            scores_for_consensus.append(val)
                    except ValueError:
                        pass

            if not all_scores:
                consensus_column.append([''])
                discrepancy_column.append([''])
                continue

            # Compute consensus (most common individual score, excluding Accepted)
            if scores_for_consensus:
                consensus = Counter(scores_for_consensus).most_common(1)[0][0]
            else:
                consensus = all_scores[0]  # Fall back to Accepted if no other scores
            consensus_column.append([str(consensus)])

            # For discrepancy, compare individual on_play scores
            # Discrepancy if any two scorers disagree on the on_play result
            if len(all_scores) >= 2:
                discrepancy = len(set(all_scores)) > 1
            else:
                discrepancy = False

            discrepancy_column.append(['TRUE' if discrepancy else 'FALSE'])

        # Write both columns in bulk (2 API calls total)
        def get_col_letter(col_idx):
            return gspread.utils.rowcol_to_a1(1, col_idx + 1).rstrip('0123456789')

        consensus_letter = get_col_letter(consensus_col_idx)
        discrepancy_letter = get_col_letter(discrepancy_col_idx)

        _rate_limiter.wait_if_needed()
        sheet.update(f"{consensus_letter}1:{consensus_letter}{len(consensus_column)}",
                    consensus_column, value_input_option='USER_ENTERED')

        _rate_limiter.wait_if_needed()
        sheet.update(f"{discrepancy_letter}1:{discrepancy_letter}{len(discrepancy_column)}",
                    discrepancy_column, value_input_option='USER_ENTERED')

        self.invalidate_cache()

    def add_scorer_column(self, scorer_name: str):
        """Add a new scorer column if it doesn't exist."""
        sheet = self._get_or_create_sheet(RESULTS_SHEET)
        headers = sheet.row_values(1)

        if scorer_name in headers:
            return  # Column already exists

        # Find position to insert (before Consensus column)
        try:
            consensus_idx = headers.index('Consensus')
        except ValueError:
            consensus_idx = len(headers)

        # Insert new column
        _rate_limiter.wait_if_needed()
        sheet.insert_cols([[scorer_name] + [''] * (sheet.row_count - 1)], consensus_idx + 1)

        # Invalidate raw cache since structure changed
        self._results_raw_cache = None

    def write_result(self, deck1: str, deck2: str, scorer: str, result: int):
        """
        Write a single result to the Results sheet.

        Args:
            deck1: Deck on play
            deck2: Deck on draw
            scorer: Scorer's name
            result: 0=Loss, 1=Tie, 2=Win (from deck1's perspective)
        """
        # Normalize deck names
        deck1 = normalize_deck(deck1)
        deck2 = normalize_deck(deck2)

        sheet = self._get_or_create_sheet(RESULTS_SHEET)

        # Find the row for this matchup
        df = self.read_results()
        mask = (df['Deck1_OnPlay'] == deck1) & (df['Deck2_OnDraw'] == deck2)

        if not mask.any():
            raise ValueError(f"Matchup not found: {deck1} vs {deck2}")

        row_idx = mask.idxmax() + 2  # +2 for 1-indexing and header row

        # Find scorer column - use cached headers if available
        now = time.time()
        if self._headers_cache is not None and (now - self._headers_cache_time) < CACHE_TTL:
            headers = self._headers_cache
        else:
            headers = sheet.row_values(1)
            self._headers_cache = headers
            self._headers_cache_time = now

        if scorer not in headers:
            self.add_scorer_column(scorer)
            headers = sheet.row_values(1)
            self._headers_cache = headers
            self._headers_cache_time = now

        col_idx = headers.index(scorer) + 1  # 1-indexed

        # Write the result (single API call)
        _rate_limiter.wait_if_needed()
        sheet.update_cell(row_idx, col_idx, str(result))

        # Update local caches to stay in sync
        if self._results_cache is not None:
            cache_idx = mask.idxmax()
            if scorer not in self._results_cache.columns:
                self._results_cache[scorer] = ''
            self._results_cache.loc[cache_idx, scorer] = str(result)

        # Also update raw cache
        if self._results_raw_cache is not None:
            raw_row_idx = row_idx - 1  # Convert to 0-indexed
            if raw_row_idx < len(self._results_raw_cache):
                # Extend row if needed
                while len(self._results_raw_cache[raw_row_idx]) < col_idx:
                    self._results_raw_cache[raw_row_idx].append('')
                self._results_raw_cache[raw_row_idx][col_idx - 1] = str(result)

        # Update consensus and discrepancy
        self._update_consensus_discrepancy(sheet, row_idx, headers)

    def write_results_batch(self, results: List[Tuple[str, str, str, int]]):
        """
        Write multiple results to the Results sheet in a single batch.

        Args:
            results: List of (deck1, deck2, scorer, result) tuples
        """
        if not results:
            return

        sheet = self._get_or_create_sheet(RESULTS_SHEET)

        # Read results once
        df = self.read_results()

        # Get headers
        now = time.time()
        if self._headers_cache is not None and (now - self._headers_cache_time) < CACHE_TTL:
            headers = self._headers_cache
        else:
            _rate_limiter.wait_if_needed()
            headers = sheet.row_values(1)
            self._headers_cache = headers
            self._headers_cache_time = now

        # Ensure all scorers exist
        scorers_needed = set(r[2] for r in results)
        for scorer in scorers_needed:
            if scorer not in headers:
                self.add_scorer_column(scorer)
                _rate_limiter.wait_if_needed()
                headers = sheet.row_values(1)
                self._headers_cache = headers
                self._headers_cache_time = now

        # Build batch update
        updates = []
        row_indices = []  # Track which rows to update consensus for

        for deck1, deck2, scorer, result in results:
            deck1 = normalize_deck(deck1)
            deck2 = normalize_deck(deck2)

            mask = (df['Deck1_OnPlay'] == deck1) & (df['Deck2_OnDraw'] == deck2)
            if not mask.any():
                continue

            row_idx = mask.idxmax() + 2  # +2 for 1-indexing and header row
            col_idx = headers.index(scorer) + 1

            # A1 notation for this cell
            col_letter = self._col_to_letter(col_idx)
            cell_ref = f"{col_letter}{row_idx}"
            updates.append({'range': cell_ref, 'values': [[str(result)]]})
            row_indices.append(row_idx)

            # Update local caches
            if self._results_cache is not None:
                cache_idx = mask.idxmax()
                if scorer not in self._results_cache.columns:
                    self._results_cache[scorer] = ''
                self._results_cache.loc[cache_idx, scorer] = str(result)

        # Batch update all cells at once
        if updates:
            _rate_limiter.wait_if_needed()
            sheet.batch_update(updates)

            # Update consensus/discrepancy for affected rows
            for row_idx in set(row_indices):
                self._update_consensus_discrepancy(sheet, row_idx, headers)

    def _col_to_letter(self, col: int) -> str:
        """Convert 1-indexed column number to letter (1='A', 27='AA', etc.)"""
        result = ""
        while col > 0:
            col, remainder = divmod(col - 1, 26)
            result = chr(65 + remainder) + result
        return result

    def _update_consensus_discrepancy(self, sheet: gspread.Worksheet, row_idx: int, headers: List[str]):
        """Update Consensus and Discrepancy columns for a row."""
        row = sheet.row_values(row_idx)

        # Pad row if needed
        while len(row) < len(headers):
            row.append('')

        # Find scorer columns
        fixed_cols = {'Deck1_OnPlay', 'Deck2_OnDraw', 'Consensus', 'Discrepancy'}
        scorer_indices = [i for i, h in enumerate(headers) if h not in fixed_cols]

        # Get scores
        scores = []
        for idx in scorer_indices:
            if idx < len(row) and row[idx].strip():
                try:
                    scores.append(int(row[idx]))
                except ValueError:
                    pass

        # Compute consensus and discrepancy
        if scores:
            # Consensus = majority vote, or first if tie
            from collections import Counter
            consensus = Counter(scores).most_common(1)[0][0]
            discrepancy = len(set(scores)) > 1
        else:
            consensus = ''
            discrepancy = ''

        # Find column indices
        try:
            consensus_col = headers.index('Consensus') + 1
            discrepancy_col = headers.index('Discrepancy') + 1
        except ValueError:
            return

        # Update cells
        sheet.update_cell(row_idx, consensus_col, str(consensus) if consensus != '' else '')
        sheet.update_cell(row_idx, discrepancy_col, str(discrepancy).upper() if discrepancy != '' else '')

    def get_matchup_result(self, deck1: str, deck2: str, scorer: str) -> Optional[int]:
        """Get a scorer's result for a specific matchup."""
        # Normalize deck names
        deck1 = normalize_deck(deck1)
        deck2 = normalize_deck(deck2)

        df = self.read_results()
        mask = (df['Deck1_OnPlay'] == deck1) & (df['Deck2_OnDraw'] == deck2)

        if not mask.any():
            return None

        if scorer not in df.columns:
            return None

        val = df.loc[mask, scorer].values[0]
        if val == '' or pd.isna(val):
            return None

        try:
            return int(val)
        except ValueError:
            return None

    def clear_result(self, deck1: str, deck2: str, scorer: str):
        """
        Clear a scorer's result for a matchup.

        Args:
            deck1: Deck on play
            deck2: Deck on draw
            scorer: Scorer's name
        """
        # Normalize deck names
        deck1 = normalize_deck(deck1)
        deck2 = normalize_deck(deck2)

        sheet = self._get_or_create_sheet(RESULTS_SHEET)

        # Find the row for this matchup
        df = self.read_results()
        mask = (df['Deck1_OnPlay'] == deck1) & (df['Deck2_OnDraw'] == deck2)

        if not mask.any():
            return

        row_idx = mask.idxmax() + 2  # +2 for 1-indexing and header row

        # Find scorer column
        headers = sheet.row_values(1)
        if scorer not in headers:
            return

        col_idx = headers.index(scorer) + 1  # 1-indexed

        # Clear the cell
        sheet.update_cell(row_idx, col_idx, '')

        # Update local cache
        if self._results_cache is not None:
            cache_idx = mask.idxmax()
            if scorer in self._results_cache.columns:
                self._results_cache.loc[cache_idx, scorer] = ''

        # Update consensus and discrepancy
        self._update_consensus_discrepancy(sheet, row_idx, headers)

    # =========================================================================
    # Matrix Sheet Operations
    # =========================================================================

    def update_matrix_sheets(self, progress_callback=None):
        """Update both matrix sheets from Results data (optimized for minimal API calls).

        Args:
            progress_callback: Optional callback(current, total, status_text) for progress updates
        """
        if progress_callback:
            progress_callback(0, 100, "Updating consensus/discrepancy...")

        # First refresh consensus/discrepancy columns
        self._update_all_consensus_discrepancy()

        if progress_callback:
            progress_callback(20, 100, "Reading results and decks...")

        df = self.read_results()
        decks = self.read_decks()

        if df.empty or not decks:
            return

        if progress_callback:
            progress_callback(30, 100, "Loading Swift data...")

        # Load Swift data for fallback
        swift = SwiftLookup.get_instance()

        if progress_callback:
            progress_callback(40, 100, "Building matrices locally...")

        # Create matrices locally
        n = len(decks)
        deck_to_idx = {d: i for i, d in enumerate(decks)}

        matrix_onplay = [[''] * (n + 1) for _ in range(n + 1)]
        matrix_ondraw = [[''] * (n + 1) for _ in range(n + 1)]

        # Set headers
        matrix_onplay[0][0] = 'Row=OnPlay'
        matrix_ondraw[0][0] = 'Row=OnDraw'

        for i, deck in enumerate(decks):
            matrix_onplay[0][i + 1] = deck
            matrix_onplay[i + 1][0] = deck
            matrix_ondraw[0][i + 1] = deck
            matrix_ondraw[i + 1][0] = deck

        if progress_callback:
            progress_callback(50, 100, "Filling in results...")

        # Fill in results from Accepted column first, then Consensus, then Swift
        total_rows = len(df)
        for idx, (_, row) in enumerate(df.iterrows()):
            if progress_callback and idx % 1000 == 0:
                pct = 50 + int((idx / total_rows) * 20)
                progress_callback(pct, 100, f"Processing matchup {idx}/{total_rows}...")

            d1 = row.get('Deck1_OnPlay', '')
            d2 = row.get('Deck2_OnDraw', '')

            # Use Accepted if available, otherwise use Consensus
            result = row.get('Accepted', '')
            if result == '':
                result = row.get('Consensus', '')

            # Fall back to Swift if no Accepted or Consensus
            if result == '':
                swift_result = swift.lookup_on_play(d1, d2)
                if swift_result is not None:
                    result = str(swift_result)

            if d1 in deck_to_idx and d2 in deck_to_idx and result != '':
                i = deck_to_idx[d1]
                j = deck_to_idx[d2]

                # Matrix_OnPlay: row is on play
                matrix_onplay[i + 1][j + 1] = str(result)

                # Matrix_OnDraw: row is on draw (so d2 vs d1, result inverted)
                try:
                    inv_result = {0: 2, 1: 1, 2: 0}[int(result)]
                    matrix_ondraw[j + 1][i + 1] = str(inv_result)
                except (ValueError, KeyError):
                    pass

        if progress_callback:
            progress_callback(75, 100, "Writing Matrix_OnPlay to sheet...")

        # Write both matrices (2 API calls total - no clear needed, update overwrites)
        sheet_onplay = self._get_or_create_sheet(MATRIX_ONPLAY_SHEET, rows=n+10, cols=n+10)
        sheet_ondraw = self._get_or_create_sheet(MATRIX_ONDRAW_SHEET, rows=n+10, cols=n+10)

        _rate_limiter.wait_if_needed()
        sheet_onplay.update('A1', matrix_onplay, value_input_option='USER_ENTERED')

        if progress_callback:
            progress_callback(90, 100, "Writing Matrix_OnDraw to sheet...")

        _rate_limiter.wait_if_needed()
        sheet_ondraw.update('A1', matrix_ondraw, value_input_option='USER_ENTERED')

        if progress_callback:
            progress_callback(100, 100, "Done!")

        self.invalidate_cache()
        return 0  # Skip validation for speed

    def validate_play_vs_draw(self) -> int:
        """
        Check that a deck on play does at least as well as on draw vs same opponent.
        Flags discrepancies where this is violated.

        Returns:
            Number of play vs draw discrepancies found.
        """
        df = self.read_results(force_refresh=True)
        if df.empty:
            return 0

        sheet = self._get_or_create_sheet(RESULTS_SHEET)
        headers = sheet.row_values(1)

        # Find Discrepancy column
        try:
            discrepancy_col = headers.index('Discrepancy') + 1
        except ValueError:
            return 0

        discrepancy_count = 0

        # Build lookup for results: (deck1_on_play, deck2_on_draw) -> result
        results_lookup = {}
        for idx, row in df.iterrows():
            d1 = row.get('Deck1_OnPlay', '')
            d2 = row.get('Deck2_OnDraw', '')
            # Use Accepted if available, otherwise Consensus
            result = row.get('Accepted', '')
            if result == '':
                result = row.get('Consensus', '')
            if d1 and d2 and result != '':
                try:
                    results_lookup[(d1, d2)] = int(result)
                except ValueError:
                    pass

        # Check each matchup
        updates = []
        for idx, row in df.iterrows():
            d1 = row.get('Deck1_OnPlay', '')
            d2 = row.get('Deck2_OnDraw', '')

            if not d1 or not d2:
                continue

            row_idx = idx + 2  # +2 for 1-indexing and header

            # Get d1's result on play vs d2
            result_on_play = results_lookup.get((d1, d2))

            # Get d1's result on draw vs d2
            # This is the inverse of d2's result on play vs d1
            d2_on_play_vs_d1 = results_lookup.get((d2, d1))

            if result_on_play is not None and d2_on_play_vs_d1 is not None:
                # d1's result on draw = 2 - d2's result on play
                result_on_draw = 2 - d2_on_play_vs_d1

                # Check: on play should be >= on draw
                if result_on_play < result_on_draw:
                    # Flag as discrepancy
                    discrepancy_count += 1
                    updates.append({
                        'range': gspread.utils.rowcol_to_a1(row_idx, discrepancy_col),
                        'values': [['TRUE']]
                    })
                else:
                    # Check if currently marked as discrepancy due to this check
                    # Only clear if no other discrepancy exists
                    current_discrepancy = row.get('Discrepancy', '')
                    if current_discrepancy == 'TRUE':
                        # Check for scorer disagreement
                        scorer_cols = self.get_scorer_columns()
                        scores = []
                        for col in scorer_cols:
                            val = row.get(col, '')
                            if val != '':
                                try:
                                    scores.append(int(val))
                                except ValueError:
                                    pass
                        has_scorer_discrepancy = len(set(scores)) > 1 if len(scores) > 1 else False

                        if not has_scorer_discrepancy:
                            updates.append({
                                'range': gspread.utils.rowcol_to_a1(row_idx, discrepancy_col),
                                'values': [['FALSE']]
                            })

        if updates:
            self._chunked_batch_update(sheet, updates)
            self.invalidate_cache()

        return discrepancy_count

    # =========================================================================
    # Nash Sheet Operations
    # =========================================================================

    def read_nash_data(self) -> Optional[pd.DataFrame]:
        """Read Nash equilibrium data from Nash sheet.

        Returns:
            DataFrame with columns: Deck, Nash Weight (%), Expected Value
            or None if no data exists.
        """
        try:
            sheet = self._get_or_create_sheet(NASH_SHEET)
            _rate_limiter.wait_if_needed()
            data = sheet.get_all_values()

            if not data or len(data) < 2:
                return None

            headers = data[0]
            rows = data[1:]
            df = pd.DataFrame(rows, columns=headers)

            # Convert numeric columns
            if 'Nash Weight (%)' in df.columns:
                df['Nash Weight (%)'] = pd.to_numeric(df['Nash Weight (%)'], errors='coerce')
            if 'Expected Value' in df.columns:
                df['Expected Value'] = pd.to_numeric(df['Expected Value'], errors='coerce')

            return df
        except Exception:
            return None

    def update_ev_vs_nash(self) -> Dict:
        """
        Update Expected Values for all decks against existing Nash weights.
        Does NOT recalculate Nash weights.

        Returns:
            Dict with 'expected_values' mapping deck -> EV
        """
        import numpy as np

        # Read existing Nash data
        nash_df = self.read_nash_data()
        if nash_df is None or nash_df.empty:
            raise ValueError("No Nash data found. Run Calculate Nash first.")

        # Build Nash weights dict from existing data
        nash_weights = {}
        for _, row in nash_df.iterrows():
            deck = row['Deck']
            weight = row['Nash Weight (%)'] / 100.0  # Convert from percentage
            nash_weights[deck] = weight

        # Build payoff matrix
        decks, payoff_matrix, missing_mask = self.build_payoff_matrix()

        # Build strategy vector in same order as decks
        strategy = np.array([nash_weights.get(deck, 0) for deck in decks])

        # Renormalize in case weights don't sum to 1
        if strategy.sum() > 0:
            strategy = strategy / strategy.sum()

        # Calculate expected value for each deck against Nash
        expected_values = {}
        for i, deck in enumerate(decks):
            ev = payoff_matrix[i] @ strategy
            expected_values[deck] = ev

        # Update the Nash sheet with new EVs (keeping existing weights)
        self.update_nash_sheet(nash_weights, expected_values)

        return {
            'expected_values': expected_values
        }

    def calculate_potential_ev(self, pending_scores: List[Tuple[str, str, str, int]] = None) -> List[Tuple[str, float, float, int]]:
        """
        Calculate the maximum potential EV for non-Nash decks.

        For each non-Nash deck, assumes all unscored matchups against Nash are wins.

        Args:
            pending_scores: Optional list of (deck1, deck2, scorer, result) tuples
                           to include in the calculation before they're saved.

        Returns:
            List of (deck, current_ev, potential_ev, unscored_count) tuples,
            sorted by potential_ev descending.
        """
        import numpy as np

        # Read existing Nash data
        nash_df = self.read_nash_data()
        if nash_df is None or nash_df.empty:
            return []

        # Get Nash decks and weights
        nash_decks = set()
        nash_weights = {}
        for _, row in nash_df.iterrows():
            deck = row['Deck']
            weight = row['Nash Weight (%)'] / 100.0
            if weight > 0.001:  # Only consider decks with meaningful weight
                nash_decks.add(deck)
                nash_weights[deck] = weight

        # Renormalize weights to sum to 1
        total_weight = sum(nash_weights.values())
        if total_weight > 0:
            nash_weights = {d: w/total_weight for d, w in nash_weights.items()}

        # Get all decks
        all_decks = set(self.read_decks())
        non_nash_decks = all_decks - nash_decks

        # Read results
        df = self.read_results()

        # Build payoff matrix for current results
        decks_list, payoff_matrix, missing_mask = self.build_payoff_matrix()
        deck_to_idx = {d: i for i, d in enumerate(decks_list)}

        # Apply pending scores to the payoff matrix and missing mask
        if pending_scores:
            for deck1, deck2, scorer, result in pending_scores:
                deck1 = normalize_deck(deck1)
                deck2 = normalize_deck(deck2)
                if deck1 in deck_to_idx and deck2 in deck_to_idx:
                    idx1 = deck_to_idx[deck1]
                    idx2 = deck_to_idx[deck2]
                    # Convert result (0=loss, 1=tie, 2=win) to payoff (-1, 0, +1)
                    payoff = result - 1  # 0->-1, 1->0, 2->+1
                    payoff_matrix[idx1, idx2] = payoff
                    payoff_matrix[idx2, idx1] = -payoff  # Inverse for other player
                    missing_mask[idx1, idx2] = False
                    missing_mask[idx2, idx1] = False

        # Build Nash strategy vector
        strategy = np.array([nash_weights.get(deck, 0) for deck in decks_list])
        if strategy.sum() > 0:
            strategy = strategy / strategy.sum()

        results = []

        for deck in non_nash_decks:
            if deck not in deck_to_idx:
                continue

            deck_idx = deck_to_idx[deck]

            # Current EV (with actual results, missing = 0)
            current_ev = payoff_matrix[deck_idx] @ strategy

            # Potential EV (missing vs Nash = 1, i.e. win)
            potential_row = payoff_matrix[deck_idx].copy()
            unscored_count = 0

            for nash_deck in nash_decks:
                if nash_deck not in deck_to_idx:
                    continue
                nash_idx = deck_to_idx[nash_deck]

                # Check if this matchup is missing/unscored
                if missing_mask[deck_idx, nash_idx]:
                    potential_row[nash_idx] = 1.0  # Assume win
                    unscored_count += 1

            potential_ev = potential_row @ strategy

            results.append((deck, current_ev, potential_ev, unscored_count))

        # Sort by potential EV descending
        results.sort(key=lambda x: -x[2])

        return results

    def update_nash_sheet(self, nash_weights: Dict[str, float], expected_values: Dict[str, float]):
        """Update Nash equilibrium sheet."""
        sheet = self._get_or_create_sheet(NASH_SHEET)

        # Sort by weight descending
        sorted_decks = sorted(nash_weights.keys(), key=lambda d: -nash_weights[d])

        data = [['Deck', 'Nash Weight (%)', 'Expected Value']]
        for deck in sorted_decks:
            weight = nash_weights[deck] * 100
            ev = expected_values.get(deck, 0)
            data.append([deck, f'{weight:.2f}', f'{ev:.4f}'])

        sheet.clear()
        sheet.update('A1', data)

    def build_payoff_matrix(self) -> Tuple[List[str], 'np.ndarray']:
        """
        Build payoff matrix by adding Matrix_OnPlay + Matrix_OnDraw.

        Returns:
            (deck_names, payoff_matrix) where payoff_matrix[i,j] is the
            sum of on-play and on-draw results for deck i vs deck j.
            Diagonal entries are set to 2 (tie equivalent).
        """
        import numpy as np

        decks = self.read_decks()
        if not decks:
            raise ValueError("No decks found")

        n = len(decks)
        deck_to_idx = {d: i for i, d in enumerate(decks)}

        # Read both matrix sheets
        try:
            sheet_onplay = self._get_or_create_sheet(MATRIX_ONPLAY_SHEET)
            sheet_ondraw = self._get_or_create_sheet(MATRIX_ONDRAW_SHEET)
            data_onplay = sheet_onplay.get_all_values()
            data_ondraw = sheet_ondraw.get_all_values()
        except Exception as e:
            raise ValueError(f"Could not read matrix sheets: {e}")

        # Initialize matrices with NaN
        matrix_onplay = np.full((n, n), np.nan)
        matrix_ondraw = np.full((n, n), np.nan)

        # Parse Matrix_OnPlay (row = on play, col = on draw)
        if len(data_onplay) > 1:
            headers = data_onplay[0][1:]  # Skip first cell
            header_to_idx = {h: i for i, h in enumerate(headers) if h in deck_to_idx}
            for row in data_onplay[1:]:
                if not row:
                    continue
                row_deck = row[0]
                if row_deck not in deck_to_idx:
                    continue
                i = deck_to_idx[row_deck]
                for col_name, j in header_to_idx.items():
                    if j + 1 < len(row) and row[j + 1].strip():
                        try:
                            matrix_onplay[i, deck_to_idx[col_name]] = float(row[j + 1])
                        except ValueError:
                            pass

        # Parse Matrix_OnDraw (row = on draw, col = on play)
        if len(data_ondraw) > 1:
            headers = data_ondraw[0][1:]  # Skip first cell
            header_to_idx = {h: i for i, h in enumerate(headers) if h in deck_to_idx}
            for row in data_ondraw[1:]:
                if not row:
                    continue
                row_deck = row[0]
                if row_deck not in deck_to_idx:
                    continue
                i = deck_to_idx[row_deck]
                for col_name, j in header_to_idx.items():
                    if j + 1 < len(row) and row[j + 1].strip():
                        try:
                            matrix_ondraw[i, deck_to_idx[col_name]] = float(row[j + 1])
                        except ValueError:
                            pass

        # Build combined payoff matrix: OnPlay[i,j] + OnDraw[i,j]
        # OnPlay[i,j] = result when i is on play vs j
        # OnDraw[i,j] = result when i is on draw vs j
        payoff_matrix = np.zeros((n, n))
        missing_mask = np.zeros((n, n), dtype=bool)  # Track which cells are missing data
        missing_count = 0

        for i in range(n):
            for j in range(n):
                if i == j:
                    payoff_matrix[i, j] = 2  # Diagonal = 2 (tie equivalent)
                else:
                    # Add on-play result (i on play vs j)
                    on_play = matrix_onplay[i, j] if not np.isnan(matrix_onplay[i, j]) else None
                    # Add on-draw result (i on draw vs j) - need to look at ondraw[i, j]
                    on_draw = matrix_ondraw[i, j] if not np.isnan(matrix_ondraw[i, j]) else None

                    if on_play is not None and on_draw is not None:
                        payoff_matrix[i, j] = on_play + on_draw
                    elif on_play is not None:
                        payoff_matrix[i, j] = on_play + 1  # Assume tie for missing
                        missing_mask[i, j] = True  # Partially missing
                    elif on_draw is not None:
                        payoff_matrix[i, j] = 1 + on_draw  # Assume tie for missing
                        missing_mask[i, j] = True  # Partially missing
                    else:
                        payoff_matrix[i, j] = 2  # Default to tie (1+1)
                        missing_mask[i, j] = True  # Fully missing
                        missing_count += 1

        if missing_count > 0:
            print(f"Warning: {missing_count} matchups have no data, defaulting to 2 (tie)")

        # Make zero-sum by subtracting 2 (so values are -2 to +2, with 0 as tie)
        payoff_matrix = payoff_matrix - 2

        return decks, payoff_matrix, missing_mask

    def update_game_matrix_sheet(self, decks: List[str], payoff_matrix: 'np.ndarray', missing_mask: 'np.ndarray' = None):
        """Save the combined game matrix (OnPlay + OnDraw) to Game_Matrix sheet with orange coloring for missing data."""
        import numpy as np

        n = len(decks)
        sheet = self._get_or_create_sheet(GAME_MATRIX_SHEET, rows=n+10, cols=n+10)

        # Build matrix data with headers
        data = [[''] + decks]  # Header row with deck names
        for i, deck in enumerate(decks):
            row = [deck] + [str(int(payoff_matrix[i, j])) for j in range(n)]
            data.append(row)

        sheet.clear()
        _rate_limiter.wait_if_needed()
        sheet.update('A1', data, value_input_option='USER_ENTERED')

    def update_decks_nash_weights(self, nash_weights: Dict[str, float]):
        """Update the Decks sheet with Nash weights column, color-coded green."""
        sheet = self._get_or_create_sheet(DECKS_SHEET)
        data = sheet.get_all_values()

        if not data:
            return

        headers = data[0]

        # Find or add Nash Weight column (column C, index 2)
        nash_col_idx = 2  # 0-indexed, so column C
        if len(headers) <= nash_col_idx or headers[nash_col_idx] != 'Nash Weight':
            # Need to ensure column C exists and has header
            if len(headers) <= nash_col_idx:
                # Extend headers
                while len(headers) <= nash_col_idx:
                    headers.append('')
            headers[nash_col_idx] = 'Nash Weight'
            sheet.update('A1', [headers])

        # Get max weight for scaling colors
        max_weight = max(nash_weights.values()) if nash_weights else 1

        # Build updates for Nash weights
        updates = []
        format_requests = []

        for i, row in enumerate(data[1:], start=2):  # Start at row 2
            if not row or not row[0].strip():
                continue

            deck_name = row[0].strip()
            # Handle HYPERLINK formulas - extract display text
            if deck_name.startswith('=HYPERLINK'):
                # Extract the display text from =HYPERLINK("url", "text")
                try:
                    deck_name = deck_name.split('"')[-2]
                except:
                    pass

            weight = nash_weights.get(deck_name, 0)
            weight_pct = weight * 100

            # Update cell value
            cell = gspread.utils.rowcol_to_a1(i, nash_col_idx + 1)  # 1-indexed
            updates.append({
                'range': cell,
                'values': [[f'{weight_pct:.2f}%']]
            })

            # Color based on weight (green gradient)
            # Higher weight = more saturated green
            intensity = weight / max_weight if max_weight > 0 else 0
            format_requests.append({
                'range': cell,
                'format': {
                    'backgroundColor': {
                        'red': 1.0 - (0.6 * intensity),
                        'green': 1.0 - (0.2 * intensity),
                        'blue': 1.0 - (0.6 * intensity)
                    }
                }
            })

        if updates:
            self._chunked_batch_update(sheet, updates)
        if format_requests:
            # Rate limit format requests too
            _rate_limiter.wait_if_needed()
            sheet.batch_format(format_requests)

    def compute_and_update_nash(self) -> Dict:
        """
        Compute Nash equilibrium from scored results and update Nash sheet.

        Returns:
            Dict with 'weights', 'expected_values', and 'game_value'
        """
        from nash import compute_nash_equilibrium

        decks, payoff_matrix, missing_mask = self.build_payoff_matrix()

        # Save the combined matrix to Game_Matrix sheet
        self.update_game_matrix_sheet(decks, payoff_matrix, missing_mask)

        # Compute Nash equilibrium
        row_strategy, col_strategy, game_value = compute_nash_equilibrium(payoff_matrix)

        # Since it's a symmetric game, row and col strategies should be similar
        # Use the average
        strategy = 0.5 * (row_strategy + col_strategy)
        strategy = strategy / strategy.sum()  # Renormalize

        # Build weights dict
        nash_weights = {deck: strategy[i] for i, deck in enumerate(decks)}

        # Calculate expected value for each deck against Nash
        expected_values = {}
        for i, deck in enumerate(decks):
            ev = payoff_matrix[i] @ strategy
            expected_values[deck] = ev

        # Update the Nash sheet
        self.update_nash_sheet(nash_weights, expected_values)

        # Update the Decks sheet with Nash weights
        self.update_decks_nash_weights(nash_weights)

        return {
            'weights': nash_weights,
            'expected_values': expected_values,
            'game_value': game_value
        }

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_scoring_stats(self, scorer: str = None) -> Dict:
        """Get statistics about scoring progress (uses cached data)."""
        df = self.read_results()  # Uses cache
        decks = self.read_decks()  # Uses cache
        history = HistoryLookup.get_instance()

        total_matchups = len(decks) * (len(decks) - 1)  # All pairs excluding mirrors

        stats = {
            'total_decks': len(decks),
            'total_matchups': total_matchups,
            'scored_matchups': 0,
            'discrepancies': 0,
        }

        if df.empty:
            return stats

        # Count scored (has consensus OR accepted)
        scored_mask = pd.Series([False] * len(df))
        if 'Consensus' in df.columns:
            scored_mask = scored_mask | (df['Consensus'] != '')
        if 'Accepted' in df.columns:
            scored_mask = scored_mask | (df['Accepted'] != '')
        stats['scored_matchups'] = scored_mask.sum()

        # Count discrepancies (including against accepted result)
        scorer_cols = self.get_scorer_columns()
        discrepancy_count = 0

        for _, row in df.iterrows():
            # Check sheet discrepancy flag
            if row.get('Discrepancy', '') == 'TRUE':
                discrepancy_count += 1
                continue

            # Check against accepted result from history
            deck1 = row.get('Deck1_OnPlay', '')
            deck2 = row.get('Deck2_OnDraw', '')
            accepted = history.lookup(deck1, deck2)

            if accepted is not None:
                for col in scorer_cols:
                    val = row.get(col, '')
                    if val != '':
                        try:
                            if int(val) != accepted:
                                discrepancy_count += 1
                                break
                        except ValueError:
                            pass

        stats['discrepancies'] = discrepancy_count

        # Count for specific scorer
        if scorer and scorer in df.columns:
            stats['scorer_count'] = (df[scorer] != '').sum()

        return stats

    def get_unscored_matchups(self, scorer: str) -> List[Tuple[str, str]]:
        """Get list of matchups not yet scored by this scorer (excludes accepted matchups)."""
        df = self.read_results()

        # Start with all matchups
        mask = pd.Series([True] * len(df))

        # Exclude matchups already scored by this scorer
        if scorer in df.columns:
            mask = mask & (df[scorer] == '')

        # Exclude matchups that have an accepted result
        if 'Accepted' in df.columns:
            mask = mask & (df['Accepted'] == '')

        unscored = df[mask]
        return list(zip(unscored['Deck1_OnPlay'], unscored['Deck2_OnDraw']))

    def get_accepted_matchups(self, scorer: str = None) -> List[Tuple[str, str]]:
        """Get list of matchups that have an accepted result."""
        df = self.read_results()

        if 'Accepted' not in df.columns:
            return []

        # Get matchups with accepted results
        mask = df['Accepted'] != ''

        # Optionally filter to only those not yet scored by this scorer
        if scorer and scorer in df.columns:
            mask = mask & (df[scorer] == '')

        accepted = df[mask]
        return list(zip(accepted['Deck1_OnPlay'], accepted['Deck2_OnDraw']))

    def get_discrepant_matchups(self) -> List[Tuple[str, str]]:
        """Get list of matchups with discrepancies (vectorized for speed)."""
        df = self.read_results()

        if 'Discrepancy' not in df.columns:
            return []

        # Fast vectorized filter - just use the Discrepancy column
        mask = df['Discrepancy'] == 'TRUE'
        discrepant = df[mask]
        return list(zip(discrepant['Deck1_OnPlay'], discrepant['Deck2_OnDraw']))

    def get_unscored_by_swift_matchups(self) -> List[Tuple[str, str]]:
        """Get list of matchups that don't have a Swift score (vectorized for speed)."""
        df = self.read_results()

        # Use Swift column in sheet if it exists
        if 'Swift' not in df.columns:
            return list(zip(df['Deck1_OnPlay'], df['Deck2_OnDraw']))

        # Fast vectorized filter
        mask = (df['Swift'] == '') & (df['Deck1_OnPlay'] != df['Deck2_OnDraw'])
        unscored = df[mask]
        return list(zip(unscored['Deck1_OnPlay'], unscored['Deck2_OnDraw']))

    def get_nash_decks(self, min_weight: float = 0.0001) -> List[str]:
        """Get list of decks that are in the current Nash equilibrium.

        Args:
            min_weight: Minimum Nash weight to be considered "in Nash" (default 0.01%)

        Returns:
            List of deck names in Nash
        """
        nash_df = self.read_nash_data()
        if nash_df is None or nash_df.empty:
            return []

        # Filter to decks with weight > min_weight
        in_nash = nash_df[nash_df['Nash Weight (%)'] > min_weight * 100]
        return in_nash['Deck'].tolist()

    def get_positive_ev_decks(self) -> List[Tuple[str, float]]:
        """Get list of decks with positive expected value vs Nash.

        Returns:
            List of (deck_name, ev) tuples for decks with EV > 0
        """
        nash_df = self.read_nash_data()
        if nash_df is None or nash_df.empty:
            return []

        positive = nash_df[nash_df['Expected Value'] > 0]
        return list(zip(positive['Deck'], positive['Expected Value']))

    def get_unscored_vs_nash_matchups(self, scorer: str, pending_scores: List[Tuple[str, str, str, int]] = None) -> List[Tuple[str, str]]:
        """Get unscored matchups involving Nash decks or decks with positive potential.

        Only includes:
        - Nash vs Nash matchups
        - Nash vs non-Nash matchups where the non-Nash deck has positive potential

        Args:
            scorer: The scorer name to check for unscored matchups
            pending_scores: Optional list of pending scores to exclude from results

        Returns:
            List of (deck1_on_play, deck2_on_draw) tuples
        """
        nash_decks = set(self.get_nash_decks())
        if not nash_decks:
            return []

        # Get decks with positive potential (including pending scores)
        potential_data = self.calculate_potential_ev(pending_scores=pending_scores)
        positive_potential_decks = set(deck for deck, curr, pot, _ in potential_data if pot > 0)

        # Relevant decks = Nash decks + positive potential decks
        relevant_decks = nash_decks | positive_potential_decks

        df = self.read_results()

        # Build mask: matchups where BOTH decks are relevant (Nash or positive potential)
        mask = (
            df['Deck1_OnPlay'].isin(relevant_decks) &
            df['Deck2_OnDraw'].isin(relevant_decks)
        )

        # Exclude already scored by this scorer
        if scorer in df.columns:
            mask = mask & (df[scorer] == '')

        # Exclude matchups with accepted results
        if 'Accepted' in df.columns:
            mask = mask & (df['Accepted'] == '')

        filtered = df[mask]
        matchups = list(zip(filtered['Deck1_OnPlay'], filtered['Deck2_OnDraw']))

        # Also exclude matchups that are in pending_scores
        if pending_scores:
            pending_set = set((normalize_deck(d1), normalize_deck(d2)) for d1, d2, _, _ in pending_scores)
            matchups = [(d1, d2) for d1, d2 in matchups if (d1, d2) not in pending_set]

        return matchups

    def compute_restricted_nash(self, extra_deck: str) -> Dict:
        """Compute Nash equilibrium for only Nash decks + one extra deck.

        This helps see if adding a single deck would change the Nash equilibrium.

        Args:
            extra_deck: The deck to add to the Nash calculation

        Returns:
            Dict with 'weights', 'game_value', 'expected_values'
        """
        import numpy as np
        from scipy.optimize import linprog

        # Get current Nash decks
        nash_decks = self.get_nash_decks()
        if not nash_decks:
            raise ValueError("No Nash decks found. Calculate Nash first.")

        # Add the extra deck
        restricted_decks = list(set(nash_decks + [extra_deck]))
        n = len(restricted_decks)
        deck_to_idx = {d: i for i, d in enumerate(restricted_decks)}

        # Build payoff matrix for restricted set
        decks, full_payoff = self.build_payoff_matrix()
        full_deck_to_idx = {d: i for i, d in enumerate(decks)}

        # Extract sub-matrix for restricted decks
        payoff = np.full((n, n), 2.0)  # Default to tie
        for i, d1 in enumerate(restricted_decks):
            for j, d2 in enumerate(restricted_decks):
                if d1 in full_deck_to_idx and d2 in full_deck_to_idx:
                    fi = full_deck_to_idx[d1]
                    fj = full_deck_to_idx[d2]
                    payoff[i, j] = full_payoff[fi, fj]

        # Normalize payoff to [-1, 1] range (original is [0, 4] for sum of two games)
        payoff_normalized = (payoff - 2) / 2

        # Solve for Nash using linear programming
        c = np.zeros(n + 1)
        c[-1] = -1  # Maximize game value

        A_ub = np.hstack([-payoff_normalized.T, np.ones((n, 1))])
        b_ub = np.zeros(n)

        A_eq = np.ones((1, n + 1))
        A_eq[0, -1] = 0
        b_eq = np.array([1.0])

        bounds = [(0, 1) for _ in range(n)] + [(None, None)]

        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

        if not result.success:
            raise ValueError(f"Nash computation failed: {result.message}")

        weights = result.x[:-1]
        game_value = result.x[-1]

        # Calculate expected values
        expected_values = {}
        for i, deck in enumerate(restricted_decks):
            ev = np.dot(payoff_normalized[i, :], weights)
            expected_values[deck] = ev

        return {
            'weights': {deck: w for deck, w in zip(restricted_decks, weights)},
            'game_value': game_value,
            'expected_values': expected_values,
            'decks': restricted_decks
        }
