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
# Scryfall Integration
# ============================================================================

@st.cache_data(ttl=3600)
def get_card_image_url(card_name: str) -> Optional[str]:
    """Fetch card image URL from Scryfall API."""
    try:
        resp = requests.get(
            "https://api.scryfall.com/cards/named",
            params={"exact": card_name},
            timeout=5
        )
        if resp.ok:
            data = resp.json()
            if 'image_uris' in data:
                return data['image_uris'].get('normal')
            elif 'card_faces' in data and data['card_faces']:
                return data['card_faces'][0].get('image_uris', {}).get('normal')
    except Exception:
        pass
    return None


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
                st.image(img_url, width=150)
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
                ['all', 'unscored', 'accepted', 'discrepancies'],
                format_func=lambda x: {
                    'all': 'All matchups',
                    'unscored': 'Unscored only (by me)',
                    'accepted': 'Accepted (from history)',
                    'discrepancies': 'Discrepancies only'
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
                st.session_state.sheets_manager.invalidate_cache()
                st.session_state.matchups = []  # Force reload
                st.session_state.need_refresh = True
                st.rerun()

            if st.button("Update Matrices"):
                try:
                    st.session_state.sheets_manager.update_accepted_column()
                    discrepancy_count = st.session_state.sheets_manager.update_matrix_sheets()
                    st.success("Matrices updated!")
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

    # Load matchups based on filter (use cached data if available)
    try:
        manager = st.session_state.sheets_manager
        scorer = st.session_state.scorer_name

        # Only load if we need to refresh or don't have matchups yet
        need_refresh = st.session_state.get('need_refresh', True)
        filter_changed = st.session_state.get('last_filter') != st.session_state.filter_mode

        if need_refresh or filter_changed or not st.session_state.matchups:
            # Reset to first matchup when switching to unscored or accepted
            if filter_changed and st.session_state.filter_mode in ('unscored', 'accepted'):
                st.session_state.current_matchup_idx = 0
                st.session_state.prev_deck_selection = '(all)'

            if st.session_state.filter_mode == 'unscored':
                matchups = manager.get_unscored_matchups(scorer)
            elif st.session_state.filter_mode == 'accepted':
                matchups = manager.get_accepted_matchups(scorer)
            elif st.session_state.filter_mode == 'discrepancies':
                matchups = manager.get_discrepant_matchups()
            else:
                df = manager.read_results()
                matchups = list(zip(df['Deck1_OnPlay'], df['Deck2_OnDraw']))

            # Sort matchups by Deck1_OnPlay so all matches for a deck are grouped
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

    # Deck dropdown for quick navigation
    deck_indices = st.session_state.get('deck_indices', {})
    if deck_indices:
        deck_list = ['(all)'] + list(deck_indices.keys())

        # Track previous selection to detect changes
        prev_deck_selection = st.session_state.get('prev_deck_selection', '(all)')

        selected_deck = st.selectbox(
            "Jump to deck (on play)",
            deck_list,
            index=0,  # Default to (all)
            key="deck_selector"
        )

        # Only jump when the selection actually changes (not on every rerun)
        if selected_deck != prev_deck_selection:
            st.session_state.prev_deck_selection = selected_deck
            if selected_deck == '(all)':
                # Jump to first matchup when selecting (all)
                st.session_state.current_matchup_idx = 0
            else:
                st.session_state.current_matchup_idx = deck_indices[selected_deck]
            st.rerun()

    # Navigation controls
    col1, col2, col3, col4 = st.columns([1, 2, 2, 1])

    with col1:
        if st.button("← Previous", disabled=idx == 0):
            st.session_state.current_matchup_idx = idx - 1
            st.rerun()

    with col2:
        new_idx = st.number_input(
            "Match #",
            min_value=1,
            max_value=total,
            value=idx + 1,
            label_visibility="collapsed"
        )
        if new_idx - 1 != idx:
            st.session_state.current_matchup_idx = new_idx - 1
            st.rerun()

    with col3:
        st.markdown(f"**of {total}**")

    with col4:
        if st.button("Next →", disabled=idx >= total - 1):
            st.session_state.current_matchup_idx = idx + 1
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
                inverse = (deck2, deck1)
                try:
                    inverse_idx = st.session_state.matchups.index(inverse)
                    st.session_state.current_matchup_idx = inverse_idx
                    st.rerun()
                except ValueError:
                    st.warning("Inverse matchup not found")

        display_deck(deck2, "Player 2 (On the Draw)", goldfish2)

    with right_col:
        # Big colored buttons with checkmark for selection
        win_sel = "→ " if existing_score == 2 else ""
        tie_sel = "→ " if existing_score == 1 else ""
        loss_sel = "→ " if existing_score == 0 else ""

        # WIN - Green circle
        if st.button(f"{win_sel}🟢 WIN", key="btn_win", use_container_width=True):
            try:
                manager.write_result(deck1, deck2, scorer, 2)
                if idx < total - 1:
                    st.session_state.current_matchup_idx = idx + 1
                st.rerun()
            except Exception as e:
                st.error(f"Failed: {e}")

        # TIE - Yellow circle
        if st.button(f"{tie_sel}🟡 TIE", key="btn_tie", use_container_width=True):
            try:
                manager.write_result(deck1, deck2, scorer, 1)
                if idx < total - 1:
                    st.session_state.current_matchup_idx = idx + 1
                st.rerun()
            except Exception as e:
                st.error(f"Failed: {e}")

        # LOSS - Red circle
        if st.button(f"{loss_sel}🔴 LOSS", key="btn_loss", use_container_width=True):
            try:
                manager.write_result(deck1, deck2, scorer, 0)
                if idx < total - 1:
                    st.session_state.current_matchup_idx = idx + 1
                st.rerun()
            except Exception as e:
                st.error(f"Failed: {e}")

        # Clear Result button
        if existing_score is not None:
            if st.button("🗑️ Clear", key="btn_clear", use_container_width=True):
                try:
                    manager.clear_result(deck1, deck2, scorer)
                    st.rerun()
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
