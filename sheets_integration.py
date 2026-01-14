"""Google Sheets integration for 3CB scoring system."""
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
from typing import List, Dict, Optional, Tuple
import os
import time
from urllib.parse import quote

# Path to history.csv (UTF-16 encoded)
HISTORY_CSV_PATH = os.path.join(os.path.dirname(__file__), 'data', 'history.csv')


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

# Cache TTL in seconds
CACHE_TTL = 30


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

        # Cache for data
        self._decks_cache = None
        self._decks_cache_time = 0
        self._results_cache = None
        self._results_cache_time = 0
        self._headers_cache = None
        self._headers_cache_time = 0
        self._goldfish_cache = None
        self._goldfish_cache_time = 0

        self._connect()

    def _connect(self):
        """Establish connection to Google Sheets."""
        if not os.path.exists(self.creds_path):
            raise FileNotFoundError(
                f"Credentials file not found: {self.creds_path}\n"
                "Please download service account credentials from Google Cloud Console."
            )

        creds = Credentials.from_service_account_file(self.creds_path, scopes=SCOPES)
        gc = gspread.authorize(creds)
        self.spreadsheet = gc.open_by_url(self.sheet_url)

    def _get_or_create_sheet(self, name: str, rows: int = 1000, cols: int = 26) -> gspread.Worksheet:
        """Get worksheet by name, creating it if it doesn't exist."""
        try:
            return self.spreadsheet.worksheet(name)
        except gspread.WorksheetNotFound:
            return self.spreadsheet.add_worksheet(title=name, rows=rows, cols=cols)

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
        self._headers_cache = None
        self._goldfish_cache = None

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
        """Update the Decks sheet so each deck name is a hyperlink to Scryfall."""
        sheet = self._get_or_create_sheet(DECKS_SHEET)
        data = sheet.get_all_values()

        if not data or len(data) < 2:
            return

        # Build updates for column A (deck names) with HYPERLINK formulas
        updates = []
        for i, row in enumerate(data):
            if i == 0:
                # Keep header as-is
                continue
            if not row or not row[0].strip():
                continue

            deck_name = row[0].strip()
            # Skip if already a formula
            if deck_name.startswith('='):
                continue

            url = self._build_scryfall_url(deck_name)
            # Google Sheets HYPERLINK formula
            formula = f'=HYPERLINK("{url}", "{deck_name}")'
            updates.append({
                'range': f'A{i + 1}',
                'values': [[formula]]
            })

        if updates:
            sheet.batch_update(updates, value_input_option='USER_ENTERED')

    def normalize_all_deck_names(self):
        """
        Normalize all deck names in Decks and Results sheets to alphabetical order.
        Updates the sheets in place.
        """
        # Normalize Decks sheet
        decks_sheet = self._get_or_create_sheet(DECKS_SHEET)
        decks_data = decks_sheet.get_all_values()

        if decks_data and len(decks_data) > 1:
            decks_updates = []
            for i, row in enumerate(decks_data[1:], start=2):
                if row and row[0].strip():
                    original = row[0].strip()
                    normalized = normalize_deck(original)
                    if original != normalized:
                        # Use HYPERLINK if it was a link, otherwise just the name
                        url = self._build_scryfall_url(normalized)
                        formula = f'=HYPERLINK("{url}", "{normalized}")'
                        decks_updates.append({
                            'range': f'A{i}',
                            'values': [[formula]]
                        })

            if decks_updates:
                decks_sheet.batch_update(decks_updates, value_input_option='USER_ENTERED')

        # Normalize Results sheet
        results_sheet = self._get_or_create_sheet(RESULTS_SHEET)
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
                results_sheet.batch_update(results_updates, value_input_option='USER_ENTERED')

        self.invalidate_cache()

    # =========================================================================
    # Results Sheet Operations
    # =========================================================================

    def read_results(self, force_refresh: bool = False) -> pd.DataFrame:
        """Read results from Results sheet (cached), with normalized deck names."""
        now = time.time()
        if not force_refresh and self._results_cache is not None and (now - self._results_cache_time) < CACHE_TTL:
            return self._results_cache.copy()

        sheet = self._get_or_create_sheet(RESULTS_SHEET)
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
        return df.copy()

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
        Creates the column if it doesn't exist.
        """
        history = HistoryLookup.get_instance()
        history.reload()  # Reload to get latest history.csv

        sheet = self._get_or_create_sheet(RESULTS_SHEET)
        headers = sheet.row_values(1)

        # Add Accepted column if it doesn't exist
        if 'Accepted' not in headers:
            # Insert before Consensus column
            try:
                consensus_idx = headers.index('Consensus')
            except ValueError:
                consensus_idx = len(headers)
            sheet.insert_cols([['Accepted'] + [''] * (sheet.row_count - 1)], consensus_idx + 1)
            headers = sheet.row_values(1)  # Refresh headers

        accepted_col_idx = headers.index('Accepted') + 1  # 1-indexed

        # Read all data
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
            sheet.batch_update(updates, value_input_option='USER_ENTERED')

        # Invalidate cache
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
        sheet.insert_cols([[scorer_name] + [''] * (sheet.row_count - 1)], consensus_idx + 1)

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

        # Write the result
        sheet.update_cell(row_idx, col_idx, str(result))

        # Update local cache
        if self._results_cache is not None:
            cache_idx = mask.idxmax()
            if scorer not in self._results_cache.columns:
                self._results_cache[scorer] = ''
            self._results_cache.loc[cache_idx, scorer] = str(result)

        # Update consensus and discrepancy
        self._update_consensus_discrepancy(sheet, row_idx, headers)

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

    def update_matrix_sheets(self):
        """Update both matrix sheets from Results data."""
        df = self.read_results()
        decks = self.read_decks()

        if df.empty or not decks:
            return

        # Create matrices
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

        # Fill in results from Accepted column first, then Consensus
        for _, row in df.iterrows():
            d1 = row.get('Deck1_OnPlay', '')
            d2 = row.get('Deck2_OnDraw', '')

            # Use Accepted if available, otherwise use Consensus
            result = row.get('Accepted', '')
            if result == '':
                result = row.get('Consensus', '')

            if d1 in deck_to_idx and d2 in deck_to_idx and result != '':
                i = deck_to_idx[d1]
                j = deck_to_idx[d2]

                # Matrix_OnPlay: row is on play
                matrix_onplay[i + 1][j + 1] = str(result)

                # Matrix_OnDraw: row is on draw (so d2 vs d1, result inverted)
                # If d1 won (2), then d2 lost (0); if d1 lost (0), d2 won (2); tie stays 1
                try:
                    inv_result = {0: 2, 1: 1, 2: 0}[int(result)]
                    matrix_ondraw[j + 1][i + 1] = str(inv_result)
                except (ValueError, KeyError):
                    pass

        # Write matrices
        sheet_onplay = self._get_or_create_sheet(MATRIX_ONPLAY_SHEET, rows=n+10, cols=n+10)
        sheet_ondraw = self._get_or_create_sheet(MATRIX_ONDRAW_SHEET, rows=n+10, cols=n+10)

        sheet_onplay.clear()
        sheet_onplay.update('A1', matrix_onplay)

        sheet_ondraw.clear()
        sheet_ondraw.update('A1', matrix_ondraw)

        # Validate play vs draw results
        return self.validate_play_vs_draw()

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
            sheet.batch_update(updates, value_input_option='USER_ENTERED')
            self.invalidate_cache()

        return discrepancy_count

    # =========================================================================
    # Nash Sheet Operations
    # =========================================================================

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
        sheet.update('A1', data)

        # Color cells orange only where data is missing (defaulted to 0/tie)
        if missing_mask is not None:
            format_requests = []
            for i in range(n):
                for j in range(n):
                    if i != j:  # Skip diagonal
                        cell_range = gspread.utils.rowcol_to_a1(i + 2, j + 2)
                        if missing_mask[i, j]:
                            # Missing data - color orange
                            format_requests.append({
                                'range': cell_range,
                                'format': {
                                    'backgroundColor': {
                                        'red': 1.0,
                                        'green': 0.8,
                                        'blue': 0.4
                                    }
                                }
                            })
                        else:
                            # Has data - white background
                            format_requests.append({
                                'range': cell_range,
                                'format': {
                                    'backgroundColor': {
                                        'red': 1.0,
                                        'green': 1.0,
                                        'blue': 1.0
                                    }
                                }
                            })

            if format_requests:
                sheet.batch_format(format_requests)

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
            sheet.batch_update(updates, value_input_option='USER_ENTERED')
        if format_requests:
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
        """Get list of matchups with discrepancies (including vs accepted result)."""
        df = self.read_results()
        history = HistoryLookup.get_instance()

        discrepant_matchups = []
        scorer_cols = self.get_scorer_columns()

        for _, row in df.iterrows():
            deck1 = row.get('Deck1_OnPlay', '')
            deck2 = row.get('Deck2_OnDraw', '')

            if not deck1 or not deck2:
                continue

            # Check if marked as discrepancy in sheet
            if row.get('Discrepancy', '') == 'TRUE':
                discrepant_matchups.append((deck1, deck2))
                continue

            # Check against accepted result from history
            accepted = history.lookup(deck1, deck2)
            if accepted is not None:
                for col in scorer_cols:
                    val = row.get(col, '')
                    if val != '' and int(val) != accepted:
                        discrepant_matchups.append((deck1, deck2))
                        break

        return discrepant_matchups
