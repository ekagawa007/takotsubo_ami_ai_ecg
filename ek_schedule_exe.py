import datetime
import time

def wait_until_target_time(target_hour, target_minute):
    now = datetime.datetime.now()
    target_time = now.replace(hour=target_hour, minute=target_minute, second=0, microsecond=0)
    
    # 指定時刻が現在時刻より前の場合、翌日の同時刻に設定
    if target_time <= now:
        target_time += datetime.timedelta(days=1)
    
    wait_seconds = (target_time - now).total_seconds()
    
    print("指定時刻まで待機します...")
    print(f"現在時刻: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"開始予定時刻: {target_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"待機時間: {wait_seconds/3600:.1f}時間")
    
    # 待機
    time.sleep(wait_seconds)
    print(f"\n開始時刻: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
