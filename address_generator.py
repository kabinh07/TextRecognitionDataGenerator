"""
Generates synthetic Bangladeshi NID-style Bangla address texts from postal_codes.csv.

Real NID address format (3 lines):
  ঠিকানা: বাসা/হোল্ডিং: গ্রামীন ব্যাংক, এরিয়া অফিস, গ্রাম/রাস্তা: হারামিয়া,
  হারামিয়া,
  ডাকঘর: বকতার হাট - ৪৩০০, সন্দ্বীপ, চট্টগ্রাম

Line count distribution: ~20% single, ~50% two-line, ~30% three-line.
"""

import csv
import random

# Bangla postal type suffixes to strip from post office names
_BN_PO_SUFFIXES = {'ইউপিও', 'ইডিবিও', 'এইচও', 'এসও', 'টিএসও', 'ইডিএসও', 'সিও', 'এমও', 'জিপিও'}

# Common locality suffixes for synthetic village name generation
_BN_LOC_SUFFIXES = [
    'পাড়া', 'বাড়ি', 'নগর', 'পুর', 'গ্রাম', 'গঞ্জ', 'হাট', 'বাজার',
    'খালী', 'তলী', 'মহল্লা', 'কান্দি', 'চর', 'ঘাট', 'তলা', 'দী', 'কাটা',
]

_BN_DIGITS = {'0': '০', '1': '১', '2': '২', '3': '৩', '4': '৪',
              '5': '৫', '6': '৬', '7': '৭', '8': '৮', '9': '৯'}

# বাসা/হোল্ডিং generation styles
_HOLDING_STYLES = ['basa', 'holding', 'num_letter', 'num_only', 'flat']
_HOLDING_WEIGHTS = [40, 20, 15, 15, 10]

_BN_LETTERS = ['ক', 'খ', 'গ', 'ঘ']


def _to_bn(n: int) -> str:
    return ''.join(_BN_DIGITS[c] for c in str(n))


def _strip_bn_suffix(name: str) -> str:
    parts = name.strip().split()
    if parts and parts[-1] in _BN_PO_SUFFIXES:
        return ' '.join(parts[:-1]).strip()
    return name.strip()


def load_postal_data(csv_path: str) -> list:
    rows = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


class AddressGenerator:
    """
    Call generate_bn() to get a single random NID Bangla address string.
    Multi-line addresses use '\\n' as separator.
    """

    def __init__(self, csv_path: str):
        self.rows = load_postal_data(csv_path)
        self._bn_villages: list = []
        self._build_village_pool()

    def _build_village_pool(self):
        bn_set: set = set()
        for row in self.rows:
            bn_po = _strip_bn_suffix(row['post_office_bn'])
            ps_bn = row['police_station_bn'].strip()
            up_bn = row.get('upokarzaloy_bn', '').strip()

            if bn_po:
                bn_set.add(bn_po)
            if up_bn:
                bn_set.add(up_bn)

            for sfx in random.sample(_BN_LOC_SUFFIXES, 4):
                if ps_bn:
                    bn_set.add(f"{ps_bn}{sfx}")
                if up_bn:
                    bn_set.add(f"{up_bn}{sfx}")

        self._bn_villages = [v for v in bn_set if v]

    # ── বাসা/হোল্ডিং value generator ─────────────────────────────────────────

    def _bn_holding(self) -> str:
        """Generate a synthetic বাসা/হোল্ডিং: field value."""
        style = random.choices(_HOLDING_STYLES, weights=_HOLDING_WEIGHTS, k=1)[0]
        num = _to_bn(random.randint(1, 500))

        if style == 'basa':
            return f"বাসা- {num}"
        elif style == 'holding':
            return f"হোল্ডিং- {num}"
        elif style == 'num_letter':
            return f"{num}/{random.choice(_BN_LETTERS)}"
        elif style == 'num_only':
            return num
        else:  # flat
            flat = _to_bn(random.randint(1, 20))
            floor_ = _to_bn(random.randint(1, 10))
            return f"ফ্ল্যাট- {flat}, তলা- {floor_}"

    # ── Main address generator ─────────────────────────────────────────────────

    def generate_bn(self) -> str:
        """
        Return a 1-3 line Bangla NID address string.

        Format:
          [PREFIX]বাসা/হোল্ডিং: [HOLDING], গ্রাম/রাস্তা: [VILLAGE],
          [VILLAGE],                          ← 3-line only
          ডাকঘর: [PO] - [PC], [THANA], [DISTRICT]
        """
        row = random.choice(self.rows)

        po      = _strip_bn_suffix(row['post_office_bn'])
        pc      = row['code_bn'] or (_to_bn(int(row['code'])) if row['code'] else '')
        thana   = row['police_station_bn']
        district = row['district_bn']
        village = random.choice(self._bn_villages)
        holding = self._bn_holding()

        # Last line is always ডাকঘর with postal code
        if pc:
            po_line = f"ডাকঘর: {po} - {pc}, {thana}, {district}"
        else:
            po_line = f"ডাকঘর: {po}, {thana}, {district}"

        fmt = random.choices([1, 2, 3], weights=[20, 50, 30])[0]

        if fmt == 1:
            # Single line — everything comma-separated
            return (
                f"বাসা/হোল্ডিং: {holding}, "
                f"গ্রাম/রাস্তা: {village}, {po_line}"
            )

        elif fmt == 2:
            # Two lines: holding+village / ডাকঘর line
            line1 = f"বাসা/হোল্ডিং: {holding}, গ্রাম/রাস্তা: {village},"
            return line1 + '\n' + po_line

        else:
            # Three lines: holding+village / village continuation / ডাকঘর line
            line1 = f"বাসা/হোল্ডিং: {holding}, গ্রাম/রাস্তা: {village},"
            line2 = f"{village},"
            return line1 + '\n' + line2 + '\n' + po_line

    def show_samples(self, n: int = 10):
        """Print n sample Bangla addresses for format review."""
        print("\n══ BANGLA ADDRESS SAMPLES ══════════════════════════════")
        for i in range(n):
            addr = self.generate_bn()
            lines = addr.split('\n')
            print(f"\n[{i+1}] ({len(lines)} line{'s' if len(lines)>1 else ''})")
            for line in lines:
                print(f"   {line}")


if __name__ == '__main__':
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'data/postal_codes.csv'
    gen = AddressGenerator(csv_path)
    print(f"Village pool: {len(gen._bn_villages):,} synthetic Bangla names")
    gen.show_samples(n=10)
