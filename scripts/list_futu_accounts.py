#!/usr/bin/env python3
"""List Futu OpenD trading accounts for acc_id/account-number mapping."""

from __future__ import annotations

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    from futu import OpenSecTradeContext, RET_OK, SecurityFirm, TrdMarket

    for firm in [SecurityFirm.FUTUSECURITIES]:
        for market in [TrdMarket.HK, TrdMarket.US]:
            ctx = OpenSecTradeContext(
                filter_trdmarket=market,
                host="127.0.0.1",
                port=11111,
                security_firm=firm,
            )
            try:
                ret, data = ctx.get_acc_list()
                print(f"\nfirm={firm} market={market} ret={ret}")
                if ret == RET_OK and hasattr(data, "to_string"):
                    print(data.to_string())
                else:
                    print(data)
            finally:
                ctx.close()


if __name__ == "__main__":
    main()
