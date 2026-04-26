"""Register the Prefect flow with a weekly schedule.

Run once locally (or via CI) after `prefect server start`:

    python scripts/deploy_prefect.py
"""

from prefect.client.schemas.schedules import CronSchedule

from worldcup2026.pipeline.flow import worldcup2026_pipeline


def main() -> None:
    worldcup2026_pipeline.serve(
        name="worldcup2026-weekly",
        schedule=CronSchedule(cron="0 6 * * MON", timezone="America/Bogota"),
        tags=["mlops", "worldcup2026"],
        description="Runs every Monday 06:00 Bogotá — pulls new FIFA results and retrains.",
    )


if __name__ == "__main__":
    main()
