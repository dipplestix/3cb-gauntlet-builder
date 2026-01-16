"""
3CB Gauntlet Builder

A Streamlit app for scoring deck matchups with Google Sheets integration.
Multiple scorers can contribute, with automatic discrepancy detection.

Run with: streamlit run app.py
"""
import streamlit as st
import requests
from typing import List, Tuple, Optional
import os
import json
import time
import pandas as pd

# Page config
st.set_page_config(
    page_title="Gauntlet Builder",
    page_icon="🎴",
    layout="wide"
)

# Custom CSS for compact layout and colored score buttons
st.markdown("""
<style>
    /* Reduce vertical spacing */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
    }
    div[data-testid="stVerticalBlock"] > div {
        gap: 0.3rem;
    }
    hr {
        margin: 0.5rem 0;
    }
    /* Compress sidebar */
    section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] > div {
        gap: 0.1rem;
    }
    section[data-testid="stSidebar"] hr {
        margin: 0.3rem 0;
    }
    section[data-testid="stSidebar"] h2 {
        font-size: 1rem;
        margin-bottom: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Scryfall Integration (with persistent cache)
# ============================================================================

CARD_CACHE_FILE = "data/card_image_cache.json"

def _load_card_cache() -> dict:
    """Load card image URLs from persistent cache file."""
    try:
        if os.path.exists(CARD_CACHE_FILE):
            with open(CARD_CACHE_FILE, 'r') as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def _save_card_cache(cache: dict):
    """Save card image URLs to persistent cache file."""
    try:
        os.makedirs(os.path.dirname(CARD_CACHE_FILE), exist_ok=True)
        with open(CARD_CACHE_FILE, 'w') as f:
            json.dump(cache, f)
    except Exception:
        pass

# Load cache into session state
if 'card_image_cache' not in st.session_state:
    st.session_state.card_image_cache = _load_card_cache()

def get_card_image_url(card_name: str) -> Optional[str]:
    """Get card image URL - from cache or Scryfall API."""
    # Check session cache first
    if card_name in st.session_state.card_image_cache:
        return st.session_state.card_image_cache[card_name]

    # Fetch from Scryfall
    try:
        resp = requests.get(
            "https://api.scryfall.com/cards/named",
            params={"exact": card_name},
            timeout=5
        )
        if resp.ok:
            data = resp.json()
            url = None
            if 'image_uris' in data:
                url = data['image_uris'].get('normal')
            elif 'card_faces' in data and data['card_faces']:
                url = data['card_faces'][0].get('image_uris', {}).get('normal')
            # Cache it
            if url:
                st.session_state.card_image_cache[card_name] = url
                _save_card_cache(st.session_state.card_image_cache)
            return url
    except Exception:
        pass
    return None

def preload_card_images(decks: list):
    """Pre-load all card images for the given decks."""
    # Get all unique cards
    all_cards = set()
    for deck in decks:
        cards = [c.strip() for c in deck.split('|')]
        all_cards.update(cards)

    # Find cards not in cache
    cache = st.session_state.card_image_cache
    missing = [c for c in all_cards if c not in cache]

    if not missing:
        return 0

    # Fetch missing cards
    for card in missing:
        get_card_image_url(card)  # This will cache it
        time.sleep(0.1)  # Rate limit Scryfall API

    return len(missing)


# ============================================================================
# Card Price Functions
# ============================================================================

PRICE_CACHE_FILE = "data/card_price_cache.json"

def _load_price_cache() -> dict:
    """Load card prices from persistent cache file."""
    try:
        if os.path.exists(PRICE_CACHE_FILE):
            with open(PRICE_CACHE_FILE, 'r') as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def _save_price_cache(cache: dict):
    """Save card prices to persistent cache file."""
    try:
        os.makedirs(os.path.dirname(PRICE_CACHE_FILE), exist_ok=True)
        with open(PRICE_CACHE_FILE, 'w') as f:
            json.dump(cache, f)
    except Exception:
        pass

if 'card_price_cache' not in st.session_state:
    st.session_state.card_price_cache = _load_price_cache()

def get_card_price(card_name: str) -> Optional[dict]:
    """Get card price - from cache or Scryfall API.

    Returns dict with 'usd', 'usd_foil', 'eur', 'tix' prices.
    """
    # Check session cache first
    if card_name in st.session_state.card_price_cache:
        return st.session_state.card_price_cache[card_name]

    # Fetch from Scryfall
    try:
        resp = requests.get(
            "https://api.scryfall.com/cards/named",
            params={"exact": card_name},
            timeout=5
        )
        if resp.ok:
            data = resp.json()
            prices = data.get('prices', {})
            price_data = {
                'usd': prices.get('usd'),
                'usd_foil': prices.get('usd_foil'),
                'eur': prices.get('eur'),
                'tix': prices.get('tix'),
                'name': data.get('name', card_name),
                'set': data.get('set_name', ''),
            }
            # Cache it
            st.session_state.card_price_cache[card_name] = price_data
            _save_price_cache(st.session_state.card_price_cache)
            return price_data
    except Exception:
        pass
    return None

def generate_price_spreadsheet(decks: list, progress_callback=None) -> str:
    """Generate a CSV spreadsheet with card prices for all cards in the gauntlet.

    Returns the path to the generated file.
    """
    import csv

    # Get all unique cards
    all_cards = set()
    card_to_decks = {}  # Track which decks each card appears in

    for deck in decks:
        cards = [c.strip() for c in deck.split('|')]
        for card in cards:
            all_cards.add(card)
            if card not in card_to_decks:
                card_to_decks[card] = []
            card_to_decks[card].append(deck)

    all_cards = sorted(all_cards)
    total = len(all_cards)

    # Fetch prices for all cards
    card_prices = []
    for i, card in enumerate(all_cards):
        if progress_callback:
            progress_callback(i, total, f"Fetching price for {card}...")

        price_data = get_card_price(card)
        if price_data:
            card_prices.append({
                'Card': price_data.get('name', card),
                'USD': price_data.get('usd') or 'N/A',
                'USD Foil': price_data.get('usd_foil') or 'N/A',
                'EUR': price_data.get('eur') or 'N/A',
                'MTGO Tix': price_data.get('tix') or 'N/A',
                'Set': price_data.get('set', ''),
                'Deck Count': len(card_to_decks.get(card, [])),
            })
        else:
            card_prices.append({
                'Card': card,
                'USD': 'N/A',
                'USD Foil': 'N/A',
                'EUR': 'N/A',
                'MTGO Tix': 'N/A',
                'Set': '',
                'Deck Count': len(card_to_decks.get(card, [])),
            })

        time.sleep(0.1)  # Rate limit Scryfall API

    # Sort by USD price (highest first), putting N/A at the end
    def sort_key(x):
        try:
            return -float(x['USD'])
        except (ValueError, TypeError):
            return 0

    card_prices.sort(key=sort_key)

    # Write to CSV
    output_path = "data/card_prices.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['Card', 'USD', 'USD Foil', 'EUR', 'MTGO Tix', 'Set', 'Deck Count'])
        writer.writeheader()
        writer.writerows(card_prices)

    if progress_callback:
        progress_callback(total, total, "Done!")

    return output_path


def display_deck(deck_name: str, label: str, goldfish: str = None, inverse_result: int = None):
    """Display a deck's cards with images."""
    # Show label with goldfish turn as a badge, and inverse result icon
    badges = []
    if goldfish:
        badges.append(f"<span style='background:#444; color:#fff; padding:2px 8px; border-radius:10px; font-size:0.85rem;'>🐟{goldfish}</span>")
    if inverse_result is not None:
        inverse_icon = {2: "🟢", 1: "🟡", 0: "🔴"}.get(inverse_result, "")
        badges.append(f"<span style='font-size:0.85rem;' title='Result when on draw'>{inverse_icon}</span>")

    if badges:
        st.markdown(f"**{label}** · {' '.join(badges)}", unsafe_allow_html=True)
    else:
        st.markdown(f"**{label}**")
    cards = [c.strip() for c in deck_name.split('|')]

    # Use padding columns to center and shrink the cards
    pad_left, *card_cols, pad_right = st.columns([1] + [1] * len(cards) + [1])

    for i, card in enumerate(cards):
        with card_cols[i]:
            img_url = get_card_image_url(card)
            if img_url:
                st.image(img_url, width=165)
            else:
                st.caption(card)


# ============================================================================
# Session State Management
# ============================================================================

def init_session_state():
    """Initialize session state variables."""
    if 'sheets_manager' not in st.session_state:
        st.session_state.sheets_manager = None
    if 'scorer_name' not in st.session_state:
        st.session_state.scorer_name = ''
    if 'current_matchup_idx' not in st.session_state:
        st.session_state.current_matchup_idx = 0
    if 'matchups' not in st.session_state:
        st.session_state.matchups = []
    if 'filter_mode' not in st.session_state:
        st.session_state.filter_mode = 'all'
    if 'connected' not in st.session_state:
        st.session_state.connected = False


# ============================================================================
# Main App
# ============================================================================

def main():
    init_session_state()

    st.title("Gauntlet Builder")

    # Sidebar for configuration
    with st.sidebar:
        # Only show configuration if not connected
        if not st.session_state.connected:
            st.header("Configuration")

            # Google Sheet URL
            sheet_url = st.text_input(
                "Google Sheet URL",
                value="https://docs.google.com/spreadsheets/d/1i9FWfoi6uyn6uL96hUJr8JY2_y5SBvhYUfxia0FE3U4/edit?usp=sharing",
                help="Paste the full URL of your Google Sheet"
            )

            # Credentials file
            creds_path = st.text_input(
                "Credentials Path",
                value="credentials.json",
                help="Path to your Google service account credentials JSON"
            )

            # Scorer name
            scorer_name = st.text_input(
                "Your Name",
                value=st.session_state.scorer_name,
                help="Your scorer name (creates a column in the sheet)"
            )
            st.session_state.scorer_name = scorer_name

            # Connect button
            if st.button("Connect to Sheet", type="primary"):
                if not sheet_url:
                    st.error("Please enter a Google Sheet URL")
                elif not scorer_name:
                    st.error("Please enter your name")
                elif not os.path.exists(creds_path):
                    st.error(f"Credentials file not found: {creds_path}")
                else:
                    try:
                        from sheets_integration import SheetsManager
                        st.session_state.sheets_manager = SheetsManager(sheet_url, creds_path)
                        st.session_state.connected = True
                        st.success("Connected!")

                        # Preload card images for all decks
                        decks = st.session_state.sheets_manager.read_decks()
                        if decks:
                            with st.spinner(f"Loading card images for {len(decks)} decks..."):
                                loaded = preload_card_images(decks)
                                if loaded > 0:
                                    st.success(f"Cached {loaded} new card images")

                        st.rerun()
                    except Exception as e:
                        st.error(f"Connection failed: {e}")

            st.divider()
        else:
            # Show connected status and scorer name
            st.markdown(f"**Scorer:** {st.session_state.scorer_name}")

        # Filter options
        if st.session_state.connected:
            st.header("Filter")
            filter_mode = st.radio(
                "Show matchups:",
                ['all', 'unscored', 'accepted', 'discrepancies', 'unscored_swift', 'unscored_vs_nash'],
                format_func=lambda x: {
                    'all': 'All matchups',
                    'unscored': 'Unscored only (by me)',
                    'accepted': 'Accepted (from history)',
                    'discrepancies': 'Discrepancies only',
                    'unscored_swift': 'Unscored by Swift',
                    'unscored_vs_nash': 'Unscored vs Nash'
                }[x]
            )
            st.session_state.filter_mode = filter_mode

            # Stats
            st.divider()
            try:
                stats = st.session_state.sheets_manager.get_scoring_stats(st.session_state.scorer_name)
                st.markdown(f"**Decks:** {stats['total_decks']}")
                st.markdown(f"**Scored:** {stats['scored_matchups']}/{stats['total_matchups']}")
                st.markdown(f"**Discrepancies:** {stats['discrepancies']}")
            except Exception as e:
                st.caption(f"Stats: {e}")

            # Actions
            st.header("Actions")

            if st.button("📥 Sync All Data"):
                """Download all data from sheets and cache locally."""
                try:
                    with st.spinner("Downloading all data..."):
                        manager = st.session_state.sheets_manager

                        # Download and cache everything
                        st.session_state.local_decks = manager.read_decks()
                        st.session_state.local_results = manager.read_results()
                        st.session_state.local_nash = manager.read_nash_data()

                        # Build and cache payoff matrix
                        decks, payoff, missing = manager.build_payoff_matrix()
                        st.session_state.local_payoff_matrix = payoff
                        st.session_state.local_missing_mask = missing
                        st.session_state.local_deck_list = decks
                        st.session_state.local_deck_to_idx = {d: i for i, d in enumerate(decks)}

                        # Cache Nash weights if available
                        if st.session_state.local_nash is not None:
                            nash_weights = {}
                            for _, row in st.session_state.local_nash.iterrows():
                                deck = row['Deck']
                                weight = row['Nash Weight (%)'] / 100.0
                                nash_weights[deck] = weight
                            st.session_state.local_nash_weights = nash_weights
                        else:
                            st.session_state.local_nash_weights = {}

                        # Clear other caches
                        st.session_state.potential_cache = None
                        st.session_state.need_refresh = True

                    st.success("All data synced locally!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to sync: {e}")

            if st.button("Refresh Data"):
                try:
                    # Normalize all deck names to alphabetical order
                    st.session_state.sheets_manager.normalize_all_deck_names()
                    # Sync any new decks to results sheet
                    new_count = st.session_state.sheets_manager.sync_results_with_decks()
                    if new_count > 0:
                        st.success(f"Added {new_count} new matchups!")
                    # Update deck links to Scryfall
                    st.session_state.sheets_manager.update_deck_links()
                    # Update Accepted column from history.csv
                    st.session_state.sheets_manager.update_accepted_column()
                except Exception as e:
                    st.warning(f"Sync issue: {e}")
                # Re-download all data from sheet
                st.session_state.sheets_manager.refresh_data()
                st.session_state.matchups = []  # Force reload
                st.session_state.need_refresh = True
                st.rerun()

            if st.button("Update Matrices"):
                try:
                    progress_bar = st.progress(0, text="Updating Accepted column from history...")

                    st.session_state.sheets_manager.update_accepted_column()

                    def update_progress(current, total, text):
                        progress_bar.progress(current / total, text=text)

                    discrepancy_count = st.session_state.sheets_manager.update_matrix_sheets(
                        progress_callback=update_progress
                    )

                    # Update EV vs Nash (without recalculating Nash weights)
                    progress_bar.progress(0.9, text="Updating performance vs Nash...")
                    try:
                        st.session_state.sheets_manager.update_ev_vs_nash()
                        progress_bar.empty()
                        st.success("Matrices updated! EVs vs Nash recalculated.")
                    except ValueError:
                        # No Nash data yet, just skip EV calculation
                        progress_bar.empty()
                        st.success("Matrices updated!")

                    # Update local cache with new data
                    progress_bar = st.progress(0.95, text="Syncing local cache...")
                    decks, payoff, missing = st.session_state.sheets_manager.build_payoff_matrix()
                    st.session_state.local_payoff_matrix = payoff
                    st.session_state.local_missing_mask = missing
                    st.session_state.local_deck_list = decks
                    st.session_state.local_deck_to_idx = {d: i for i, d in enumerate(decks)}
                    st.session_state.local_nash = st.session_state.sheets_manager.read_nash_data()
                    if st.session_state.local_nash is not None:
                        nash_weights = {}
                        for _, row in st.session_state.local_nash.iterrows():
                            nash_weights[row['Deck']] = row['Nash Weight (%)'] / 100.0
                        st.session_state.local_nash_weights = nash_weights
                    progress_bar.empty()

                    if discrepancy_count and discrepancy_count > 0:
                        st.error(f"⚠️ {discrepancy_count} play vs draw discrepancies found!")
                except Exception as e:
                    st.error(f"Failed to update matrices: {e}")

            if st.button("Calculate Nash"):
                try:
                    with st.spinner("Computing Nash equilibrium..."):
                        result = st.session_state.sheets_manager.compute_and_update_nash()
                    st.success(f"Nash computed! Game value: {result['game_value']:.4f}")
                    # Show top decks
                    weights = result['weights']
                    top_decks = sorted(weights.items(), key=lambda x: -x[1])[:5]
                    st.markdown("**Top 5 decks:**")
                    for deck, weight in top_decks:
                        if weight > 0.001:
                            st.markdown(f"- {deck}: {weight*100:.1f}%")

                    # Update local cache with new Nash data
                    st.session_state.local_nash = st.session_state.sheets_manager.read_nash_data()
                    st.session_state.local_nash_weights = weights

                    # Also refresh payoff matrix
                    decks, payoff, missing = st.session_state.sheets_manager.build_payoff_matrix()
                    st.session_state.local_payoff_matrix = payoff
                    st.session_state.local_missing_mask = missing
                    st.session_state.local_deck_list = decks
                    st.session_state.local_deck_to_idx = {d: i for i, d in enumerate(decks)}

                except Exception as e:
                    st.error(f"Failed to compute Nash: {e}")

            if st.button("Initialize Results Sheet"):
                try:
                    decks = st.session_state.sheets_manager.read_decks()
                    if not decks:
                        st.error("No decks found in Decks sheet")
                    else:
                        st.session_state.sheets_manager.initialize_results_sheet(decks)
                        st.success(f"Initialized with {len(decks)} decks!")
                        st.rerun()
                except Exception as e:
                    st.error(f"Failed to initialize: {e}")

            if st.button("Import Swift Scores"):
                try:
                    progress_bar = st.progress(0, text="Starting import...")
                    status_text = st.empty()

                    def update_progress(current, total, text):
                        progress_bar.progress(current / total, text=text)
                        status_text.text(text)

                    counts = st.session_state.sheets_manager.import_swift_scores(
                        progress_callback=update_progress
                    )
                    progress_bar.empty()
                    status_text.empty()

                    st.success(f"Imported {counts['imported']} new scores!")
                    if counts.get('skipped_existing', 0) > 0:
                        st.info(f"Skipped {counts['skipped_existing']} already populated")
                    if counts.get('skipped_ambiguous', 0) > 0:
                        st.info(f"Skipped {counts['skipped_ambiguous']} ambiguous (total=2) matchups")
                    if counts['skipped_missing'] > 0:
                        st.warning(f"Missing Swift data for {counts['skipped_missing']} matchups")
                    st.session_state.sheets_manager.invalidate_cache()
                    st.session_state.matchups = []
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to import Swift: {e}")


    # Main content
    if not st.session_state.connected:
        st.info("Please connect to a Google Sheet using the sidebar.")
        st.markdown("""
        ### Setup Instructions

        1. **Create a Google Cloud Project** and enable the Google Sheets API
        2. **Create a Service Account** and download the credentials JSON
        3. **Create a Google Sheet** with these tabs:
           - `Decks` - List your deck names in column A (e.g., "Card1 | Card2 | Card3")
           - `Matrix_OnPlay` - Will be auto-populated
           - `Matrix_OnDraw` - Will be auto-populated
           - `Nash` - Will be auto-populated
           - `Results` - Will be auto-populated
        4. **Share the Sheet** with your service account email (found in credentials JSON)
        5. **Enter the Sheet URL** and your name in the sidebar, then click Connect
        """)
        return

    # Create tabs
    tab_scoring, tab_nash, tab_potential = st.tabs(["Scoring", "Performance vs Nash", "Potential"])

    # Performance vs Nash tab
    with tab_nash:
        st.header("Performance vs Nash")

        # Use local cached Nash data if available
        nash_df = st.session_state.get('local_nash', None)

        if nash_df is None or (hasattr(nash_df, 'empty') and nash_df.empty):
            st.warning("No Nash data. Click '📥 Sync All Data' after running 'Calculate Nash'.")
        else:
            # Sort options
            sort_col = st.selectbox(
                "Sort by",
                ["Expected Value", "Nash Weight (%)", "Deck"],
                index=0
            )
            sort_asc = st.checkbox("Ascending", value=False)

            # Ensure numeric columns are actually numeric before sorting
            nash_df = nash_df.copy()
            nash_df['Nash Weight (%)'] = pd.to_numeric(nash_df['Nash Weight (%)'], errors='coerce').fillna(0)
            nash_df['Expected Value'] = pd.to_numeric(nash_df['Expected Value'], errors='coerce').fillna(0)

            # Sort the dataframe
            nash_df_sorted = nash_df.sort_values(by=sort_col, ascending=sort_asc)

            # Format for display
            display_df = nash_df_sorted.copy()
            display_df['Nash Weight (%)'] = display_df['Nash Weight (%)'].apply(lambda x: f"{x:.2f}%")
            display_df['Expected Value'] = display_df['Expected Value'].apply(lambda x: f"{x:.4f}")

            # Display as table
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                height=600
            )

            # Summary stats
            st.divider()
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Decks", len(nash_df))
            with col2:
                positive_ev = (nash_df['Expected Value'] > 0).sum()
                st.metric("Positive EV Decks", positive_ev)
            with col3:
                in_nash = (nash_df['Nash Weight (%)'] > 0.01).sum()
                st.metric("In Nash (>0.01%)", in_nash)

            # Restricted Nash calculation
            st.divider()
            st.subheader("Calculate Restricted Nash")

            # Get decks with positive EV from local data
            positive_ev_decks = [(row['Deck'], row['Expected Value'])
                                 for _, row in nash_df.iterrows()
                                 if row['Expected Value'] > 0]
            positive_ev_decks.sort(key=lambda x: -x[1])

            if not positive_ev_decks:
                st.info("No decks with positive EV vs Nash. Nothing to test.")
            else:
                # Create selection dropdown for positive EV decks
                deck_options = [f"{deck} (EV: {ev:.4f})" for deck, ev in positive_ev_decks]
                deck_names = [deck for deck, ev in positive_ev_decks]

                selected_idx = st.selectbox(
                    "Select deck to test",
                    range(len(deck_options)),
                    format_func=lambda i: deck_options[i],
                    key="restricted_nash_deck"
                )

                selected_deck = deck_names[selected_idx]

                if st.button("Calculate Restricted Nash", key="btn_restricted_nash"):
                    try:
                        with st.spinner(f"Computing restricted Nash with {selected_deck}..."):
                            result = st.session_state.sheets_manager.compute_restricted_nash(selected_deck)

                        st.success(f"Restricted Nash computed! Game value: {result['game_value']:.4f}")

                        # Show results
                        st.markdown(f"**Decks in calculation:** {len(result['decks'])}")

                        # Show weights table
                        weights_data = []
                        for deck in result['decks']:
                            weight = result['weights'].get(deck, 0)
                            ev = result['expected_values'].get(deck, 0)
                            weights_data.append({
                                'Deck': deck,
                                'New Weight (%)': f"{weight*100:.2f}%",
                                'New EV': f"{ev:.4f}"
                            })

                        # Sort by weight
                        weights_data.sort(key=lambda x: -float(x['New Weight (%)'].replace('%', '')))

                        st.dataframe(weights_data, use_container_width=True, hide_index=True)

                        # Check if the selected deck enters Nash
                        new_weight = result['weights'].get(selected_deck, 0)
                        if new_weight > 0.0001:
                            st.success(f"**{selected_deck}** enters Nash with {new_weight*100:.2f}% weight!")
                        else:
                            st.warning(f"**{selected_deck}** does NOT enter Nash in restricted calculation.")

                    except Exception as e:
                        st.error(f"Failed to compute restricted Nash: {e}")

    # Potential tab
    with tab_potential:
        st.header("Potential EV")
        st.markdown("""
        Shows the **maximum potential EV** for non-Nash decks, assuming all unscored
        matchups against Nash decks are wins. Uses locally cached data.
        """)

        # Check if we have local data
        if 'local_payoff_matrix' not in st.session_state or st.session_state.local_payoff_matrix is None:
            st.warning("No local data. Click '📥 Sync All Data' in the sidebar first.")
        elif 'local_nash_weights' not in st.session_state or not st.session_state.local_nash_weights:
            st.warning("No Nash data. Run 'Calculate Nash' first, then sync.")
        else:
            try:
                import numpy as np

                # Use local cached data
                payoff_matrix = st.session_state.local_payoff_matrix.copy()
                missing_mask = st.session_state.local_missing_mask.copy()
                decks_list = st.session_state.local_deck_list
                deck_to_idx = st.session_state.local_deck_to_idx
                nash_weights = st.session_state.local_nash_weights
                all_decks = set(st.session_state.local_decks) if 'local_decks' in st.session_state else set(decks_list)

                # Get Nash decks (weight > 0.1%)
                nash_decks = set(d for d, w in nash_weights.items() if w > 0.001)
                non_nash_decks = all_decks - nash_decks

                # Apply pending scores to local copy
                pending = st.session_state.get('pending_scores', [])
                if pending:
                    for deck1, deck2, _, result in pending:
                        if deck1 in deck_to_idx and deck2 in deck_to_idx:
                            idx1 = deck_to_idx[deck1]
                            idx2 = deck_to_idx[deck2]
                            payoff = result - 1  # 0->-1, 1->0, 2->+1
                            payoff_matrix[idx1, idx2] = payoff
                            payoff_matrix[idx2, idx1] = -payoff
                            missing_mask[idx1, idx2] = False
                            missing_mask[idx2, idx1] = False
                    st.info(f"📝 Including {len(pending)} pending score(s)")

                # Build Nash strategy vector
                strategy = np.array([nash_weights.get(deck, 0) for deck in decks_list])
                if strategy.sum() > 0:
                    strategy = strategy / strategy.sum()

                # Calculate potential for each non-Nash deck
                potential_data = []
                for deck in non_nash_decks:
                    if deck not in deck_to_idx:
                        continue

                    deck_idx = deck_to_idx[deck]
                    current_ev = payoff_matrix[deck_idx] @ strategy

                    # Potential EV (missing vs Nash = win)
                    potential_row = payoff_matrix[deck_idx].copy()
                    unscored_count = 0

                    for nash_deck in nash_decks:
                        if nash_deck not in deck_to_idx:
                            continue
                        nash_idx = deck_to_idx[nash_deck]
                        if missing_mask[deck_idx, nash_idx]:
                            potential_row[nash_idx] = 1.0
                            unscored_count += 1

                    potential_ev = potential_row @ strategy
                    potential_data.append((deck, current_ev, potential_ev, unscored_count))

                # Sort by potential descending
                potential_data.sort(key=lambda x: -x[2])

                # Filter options
                show_positive_only = st.checkbox("Show only decks with positive potential", value=True)
                if show_positive_only:
                    potential_data = [p for p in potential_data if p[2] > 0]

                if not potential_data:
                    st.info("No decks with positive potential EV.")
                else:
                    display_data = []
                    for deck, current_ev, potential_ev, unscored in potential_data:
                        display_data.append({
                            'Deck': deck,
                            'Current EV': f"{current_ev:.4f}",
                            'Potential EV': f"{potential_ev:.4f}",
                            'Unscored vs Nash': unscored
                        })

                    st.dataframe(display_data, use_container_width=True, hide_index=True, height=600)

                    st.divider()
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Decks Shown", len(display_data))
                    with col2:
                        if display_data:
                            st.metric("Top Potential EV", display_data[0]['Potential EV'])

            except Exception as e:
                st.error(f"Failed to calculate potential: {e}")

    # Scoring tab
    with tab_scoring:
        # Load matchups based on filter (use cached data if available)
        try:
            manager = st.session_state.sheets_manager
            scorer = st.session_state.scorer_name

            # Only load if we need to refresh or don't have matchups yet
            need_refresh = st.session_state.get('need_refresh', True)
            filter_changed = st.session_state.get('last_filter') != st.session_state.filter_mode

            if need_refresh or filter_changed or not st.session_state.matchups:
                # Reset to first matchup when filter changes
                if filter_changed:
                    st.session_state.current_matchup_idx = 0
                    st.session_state._sync_match_number = True
                    st.session_state.deck_selector = '(all)'

                if st.session_state.filter_mode == 'unscored':
                    matchups = manager.get_unscored_matchups(scorer)
                elif st.session_state.filter_mode == 'accepted':
                    matchups = manager.get_accepted_matchups(scorer)
                elif st.session_state.filter_mode == 'discrepancies':
                    matchups = manager.get_discrepant_matchups()
                elif st.session_state.filter_mode == 'unscored_swift':
                    matchups = manager.get_unscored_by_swift_matchups()
                elif st.session_state.filter_mode == 'unscored_vs_nash':
                    # Use local data for this filter to avoid API calls
                    if 'local_nash_weights' in st.session_state and st.session_state.local_nash_weights:
                        import numpy as np
                        pending = st.session_state.get('pending_scores', [])
                        nash_weights = st.session_state.local_nash_weights
                        nash_decks = set(d for d, w in nash_weights.items() if w > 0.001)

                        # Calculate positive potential decks locally
                        if 'local_payoff_matrix' in st.session_state:
                            payoff = st.session_state.local_payoff_matrix.copy()
                            missing = st.session_state.local_missing_mask.copy()
                            decks_list = st.session_state.local_deck_list
                            deck_to_idx = st.session_state.local_deck_to_idx
                            all_decks = set(st.session_state.get('local_decks', decks_list))

                            # Apply pending scores
                            for d1, d2, _, result in pending:
                                if d1 in deck_to_idx and d2 in deck_to_idx:
                                    idx1, idx2 = deck_to_idx[d1], deck_to_idx[d2]
                                    payoff[idx1, idx2] = result - 1
                                    payoff[idx2, idx1] = -(result - 1)
                                    missing[idx1, idx2] = False
                                    missing[idx2, idx1] = False

                            strategy = np.array([nash_weights.get(d, 0) for d in decks_list])
                            if strategy.sum() > 0:
                                strategy = strategy / strategy.sum()

                            # Find decks with positive potential
                            positive_potential = set()
                            non_nash = all_decks - nash_decks
                            for deck in non_nash:
                                if deck not in deck_to_idx:
                                    continue
                                idx = deck_to_idx[deck]
                                pot_row = payoff[idx].copy()
                                for nd in nash_decks:
                                    if nd in deck_to_idx and missing[idx, deck_to_idx[nd]]:
                                        pot_row[deck_to_idx[nd]] = 1.0
                                if (pot_row @ strategy) > 0:
                                    positive_potential.add(deck)

                            relevant = nash_decks | positive_potential
                            pending_set = set((d1, d2) for d1, d2, _, _ in pending)

                            # Get unscored matchups from local results
                            df = st.session_state.get('local_results')
                            if df is not None:
                                mask = df['Deck1_OnPlay'].isin(relevant) & df['Deck2_OnDraw'].isin(relevant)
                                if scorer in df.columns:
                                    mask = mask & (df[scorer] == '')
                                if 'Accepted' in df.columns:
                                    mask = mask & (df['Accepted'] == '')
                                matchups = [(r['Deck1_OnPlay'], r['Deck2_OnDraw'])
                                           for _, r in df[mask].iterrows()
                                           if (r['Deck1_OnPlay'], r['Deck2_OnDraw']) not in pending_set]
                            else:
                                matchups = manager.get_unscored_vs_nash_matchups(scorer, pending_scores=pending)

                            # Store for custom sorting
                            st.session_state._potential_decks = positive_potential
                            st.session_state._nash_weights_sort = nash_weights
                        else:
                            matchups = manager.get_unscored_vs_nash_matchups(scorer, pending_scores=pending)
                            st.session_state._potential_decks = set()
                            st.session_state._nash_weights_sort = nash_weights
                    else:
                        matchups = manager.get_unscored_vs_nash_matchups(scorer, pending_scores=pending)
                        st.session_state._potential_decks = set()
                        st.session_state._nash_weights_sort = {}
                else:
                    df = manager.read_results()
                    matchups = list(zip(df['Deck1_OnPlay'], df['Deck2_OnDraw']))

                # Sort matchups
                if st.session_state.filter_mode == 'unscored_vs_nash':
                    # Custom sort: potential decks first, opponents by Nash weight (highest first)
                    potential_decks = st.session_state.get('_potential_decks', set())
                    nash_weights_sort = st.session_state.get('_nash_weights_sort', {})

                    def unscored_vs_nash_sort_key(matchup):
                        d1, d2 = matchup
                        # Potential decks come first (0), Nash decks second (1)
                        d1_priority = 0 if d1 in potential_decks else 1
                        # Within same priority, sort d1 alphabetically
                        # Sort d2 by Nash weight descending (negate for descending)
                        d2_nash = nash_weights_sort.get(d2, 0)
                        return (d1_priority, d1, -d2_nash)

                    matchups = sorted(matchups, key=unscored_vs_nash_sort_key)
                else:
                    # Default: sort by Deck1_OnPlay alphabetically
                    matchups = sorted(matchups, key=lambda x: x[0])

                st.session_state.matchups = matchups
                st.session_state.need_refresh = False
                st.session_state.last_filter = st.session_state.filter_mode

                # Build deck index for dropdown navigation
                deck_indices = {}
                for i, (d1, d2) in enumerate(matchups):
                    if d1 not in deck_indices:
                        deck_indices[d1] = i
                st.session_state.deck_indices = deck_indices

        except Exception as e:
            st.error(f"Failed to load matchups: {e}")
            return

        if not st.session_state.matchups:
            if st.session_state.filter_mode == 'unscored':
                st.success("You've scored all matchups!")
            elif st.session_state.filter_mode == 'accepted':
                st.success("You've scored all accepted matchups!")
            elif st.session_state.filter_mode == 'discrepancies':
                st.success("No discrepancies found!")
            elif st.session_state.filter_mode == 'unscored_swift':
                st.success("All matchups have Swift scores!")
            elif st.session_state.filter_mode == 'unscored_vs_nash':
                st.success("All non-Nash vs Nash matchups are scored!")
            else:
                st.warning("No matchups found. Click 'Initialize Results Sheet' in the sidebar.")
            return

        # Navigation
        total = len(st.session_state.matchups)
        idx = st.session_state.current_matchup_idx

        # Clamp index
        if idx >= total:
            idx = total - 1
            st.session_state.current_matchup_idx = idx
        if idx < 0:
            idx = 0
            st.session_state.current_matchup_idx = idx

        # Handle pending inverse navigation FIRST (before rendering widgets)
        if 'pending_inverse' in st.session_state:
            target_on_play, target_on_draw = st.session_state.pending_inverse
            del st.session_state.pending_inverse

            # Find the inverse in current matchups
            found_idx = None
            for i, (m1, m2) in enumerate(st.session_state.matchups):
                if m1 == target_on_play and m2 == target_on_draw:
                    found_idx = i
                    break

            if found_idx is not None:
                idx = found_idx
                st.session_state.current_matchup_idx = idx
                st.session_state._sync_match_number = True
            else:
                # Try switching to 'all' filter
                df = manager.read_results()
                all_matchups = list(zip(df['Deck1_OnPlay'], df['Deck2_OnDraw']))
                all_matchups = sorted(all_matchups, key=lambda x: x[0])

                for i, (m1, m2) in enumerate(all_matchups):
                    if m1 == target_on_play and m2 == target_on_draw:
                        found_idx = i
                        break

                if found_idx is not None:
                    st.session_state.filter_mode = 'all'
                    st.session_state.matchups = all_matchups
                    st.session_state.current_matchup_idx = found_idx
                    st.session_state.need_refresh = False
                    st.session_state.last_filter = 'all'
                    st.session_state._sync_match_number = True
                    idx = found_idx
                    total = len(all_matchups)
                    # Rebuild deck indices
                    deck_indices = {}
                    for i, (d1, d2) in enumerate(all_matchups):
                        if d1 not in deck_indices:
                            deck_indices[d1] = i
                    st.session_state.deck_indices = deck_indices
                else:
                    st.error(f"Inverse matchup not found: {target_on_play} vs {target_on_draw}")

        # Deck dropdown for quick navigation
        deck_indices = st.session_state.get('deck_indices', {})
        if deck_indices:
            deck_list = ['(all)'] + list(deck_indices.keys())

            def on_deck_select():
                """Handle deck selection change - only called on user interaction."""
                selected = st.session_state.deck_selector
                if selected == '(all)':
                    st.session_state.current_matchup_idx = 0
                    st.session_state._sync_match_number = True
                elif selected in deck_indices:
                    st.session_state.current_matchup_idx = deck_indices[selected]
                    st.session_state._sync_match_number = True

            st.selectbox(
                "Jump to deck (on play)",
                deck_list,
                index=0,
                key="deck_selector",
                on_change=on_deck_select
            )

        # Navigation controls
        from streamlit_shortcuts import shortcut_button

        # Sync match_number_input with current idx BEFORE rendering widgets
        # This ensures the number input shows the correct value
        if 'match_number_input' not in st.session_state or st.session_state.get('_sync_match_number', False):
            st.session_state.match_number_input = idx + 1
            st.session_state._sync_match_number = False

        col1, col2, col3, col4 = st.columns([1, 2, 2, 1])

        with col1:
            if shortcut_button("← Previous", "4", key="btn_prev", disabled=idx == 0):
                st.session_state.current_matchup_idx = idx - 1
                st.session_state._sync_match_number = True
                st.rerun()

        with col2:
            def on_match_number_change():
                new_val = st.session_state.match_number_input
                st.session_state.current_matchup_idx = new_val - 1

            st.number_input(
                "Match #",
                min_value=1,
                max_value=total,
                label_visibility="collapsed",
                key="match_number_input",
                on_change=on_match_number_change
            )

        with col3:
            st.markdown(f"**of {total}**")

        with col4:
            if shortcut_button("Next →", "6", key="btn_next", disabled=idx >= total - 1):
                st.session_state.current_matchup_idx = idx + 1
                st.session_state._sync_match_number = True
                st.rerun()

        st.divider()

        # Current matchup
        deck1, deck2 = st.session_state.matchups[idx]

        # Check existing score
        existing_score = manager.get_matchup_result(deck1, deck2, scorer)

        # Get goldfish turns
        goldfish1 = manager.get_goldfish(deck1)
        goldfish2 = manager.get_goldfish(deck2)

        # Two-column layout: decks on left, scoring on right
        left_col, right_col = st.columns([2, 1])

        with left_col:
            # Get inverse matchup score (deck2 on play vs deck1 on draw)
            # Check scorer's result first, then fall back to accepted result
            inverse_score = manager.get_matchup_result(deck2, deck1, scorer)
            if inverse_score is None:
                # Check for accepted result
                try:
                    df = manager.read_results()
                    mask = (df['Deck1_OnPlay'] == deck2) & (df['Deck2_OnDraw'] == deck1)
                    if mask.any() and 'Accepted' in df.columns:
                        accepted_val = df.loc[mask, 'Accepted'].values[0]
                        if accepted_val != '':
                            inverse_score = int(accepted_val)
                except Exception:
                    pass
            # Invert to show how deck1 does when on draw: 2 - score
            deck1_on_draw = (2 - inverse_score) if inverse_score is not None else None

            # Display decks stacked with goldfish and inverse result for deck1
            display_deck(deck1, "Player 1 (On the Play)", goldfish1, deck1_on_draw)

            # Swap button to view inverse matchup
            vs_col1, vs_col2, vs_col3 = st.columns([2, 1, 2])
            with vs_col2:
                if st.button("⇅ vs", key="swap_matchup", help="View inverse matchup"):
                    # Store the inverse target as deck names (not index)
                    # Current: deck1 on play, deck2 on draw
                    # Inverse: deck2 on play, deck1 on draw
                    st.session_state.pending_inverse = (deck2, deck1)
                    st.session_state._sync_match_number = True
                    st.rerun()

            display_deck(deck2, "Player 2 (On the Draw)", goldfish2)

        with right_col:
            # Initialize pending scores if not exists
            if 'pending_scores' not in st.session_state:
                st.session_state.pending_scores = []

            # Check if this matchup has a pending score
            pending_for_matchup = None
            for ps in st.session_state.pending_scores:
                if ps[0] == deck1 and ps[1] == deck2:
                    pending_for_matchup = ps[3]
                    break

            # Use pending score for display if exists, otherwise use existing_score
            display_score = pending_for_matchup if pending_for_matchup is not None else existing_score

            # Big colored buttons with checkmark for selection
            win_sel = "→ " if display_score == 2 else ""
            tie_sel = "→ " if display_score == 1 else ""
            loss_sel = "→ " if display_score == 0 else ""

            from streamlit_shortcuts import shortcut_button

            def add_pending_score(result):
                # Remove any existing pending score for this matchup
                st.session_state.pending_scores = [
                    ps for ps in st.session_state.pending_scores
                    if not (ps[0] == deck1 and ps[1] == deck2)
                ]
                # Add new pending score
                st.session_state.pending_scores.append((deck1, deck2, scorer, result))

                # Force filter refresh so unscored_vs_nash updates live
                if st.session_state.filter_mode == 'unscored_vs_nash':
                    st.session_state.need_refresh = True

                # Auto-save when we have 5 pending scores
                if len(st.session_state.pending_scores) >= 5:
                    manager.write_results_batch(st.session_state.pending_scores)
                    st.session_state.pending_scores = []

            # WIN - Green circle
            if shortcut_button(f"{win_sel}🟢 WIN", "1", key="btn_win", use_container_width=True):
                add_pending_score(2)
                if idx < total - 1:
                    st.session_state.current_matchup_idx = idx + 1
                    st.session_state._sync_match_number = True
                st.rerun()

            # TIE - Yellow circle
            if shortcut_button(f"{tie_sel}🟡 TIE", "2", key="btn_tie", use_container_width=True):
                add_pending_score(1)
                if idx < total - 1:
                    st.session_state.current_matchup_idx = idx + 1
                    st.session_state._sync_match_number = True
                st.rerun()

            # LOSS - Red circle
            if shortcut_button(f"{loss_sel}🔴 LOSS", "3", key="btn_loss", use_container_width=True):
                add_pending_score(0)
                if idx < total - 1:
                    st.session_state.current_matchup_idx = idx + 1
                    st.session_state._sync_match_number = True
                st.rerun()

            # Show pending count and save button
            pending_count = len(st.session_state.pending_scores)
            if pending_count > 0:
                st.warning(f"📝 {pending_count} pending")
                if st.button("💾 Save Now", key="btn_save_pending", use_container_width=True):
                    try:
                        manager.write_results_batch(st.session_state.pending_scores)
                        st.session_state.pending_scores = []
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed: {e}")

            # Clear Result button
            if existing_score is not None:
                if st.button("🗑️ Clear", key="btn_clear", use_container_width=True):
                    try:
                        manager.clear_result(deck1, deck2, scorer)
                    except Exception as e:
                        st.error(f"Failed: {e}")

            # Always show accepted result if it exists
            try:
                df = manager.read_results()
                mask = (df['Deck1_OnPlay'] == deck1) & (df['Deck2_OnDraw'] == deck2)
                if mask.any():
                    row = df[mask].iloc[0]
                    accepted_val = row.get('Accepted', '')

                    if accepted_val != '':
                        accepted = int(accepted_val)
                        accepted_text = {0: 'Loss', 1: 'Tie', 2: 'Win'}.get(accepted, '?')
                        st.markdown(f"**Accepted:** {accepted_text}")

                        # Check if user's score matches accepted
                        if existing_score is not None and existing_score != accepted:
                            st.error(f"⚠️ Your score doesn't match accepted!")

                    # Always show Swift score if it exists
                    swift_val = row.get('Swift', '')
                    if swift_val != '':
                        swift_text = {0: 'Loss', 1: 'Tie', 2: 'Win'}.get(int(swift_val), '?')
                        st.markdown(f"**Swift:** {swift_text}")

                    # Show other scores after user has scored
                    if existing_score is not None:
                        st.divider()
                        st.markdown("**All Scores**")

                        scorer_cols = manager.get_scorer_columns()

                        has_discrepancy = False
                        for col in scorer_cols:
                            val = row.get(col, '')
                            if val != '':
                                result_text = {0: 'Loss', 1: 'Tie', 2: 'Win'}.get(int(val), val)
                                # Check against accepted result
                                if accepted_val != '' and int(val) != int(accepted_val):
                                    st.text(f"{col}: {result_text} ❌")
                                    has_discrepancy = True
                                else:
                                    st.text(f"{col}: {result_text}")

                        consensus = row.get('Consensus', '')
                        discrepancy = row.get('Discrepancy', '')

                        if consensus:
                            consensus_text = {0: 'Loss', 1: 'Tie', 2: 'Win'}.get(int(consensus), consensus)
                            st.markdown(f"**Consensus:** {consensus_text}")

                        if discrepancy == 'TRUE' or has_discrepancy:
                            st.warning("⚠️ Discrepancy!")
            except Exception as e:
                st.warning(f"Error: {e}")


if __name__ == "__main__":
    main()
