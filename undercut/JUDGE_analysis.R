library(tidyverse)
library(nflreadr)
library(ggimage)
library(gt)
library(nflfastR)
library(ggplot2)
library(ggpmisc)
library(ggsci)
library(scales)
library(ggplot2)

rosters <- load_rosters(seasons = c(2023))

rost_info_needed <- rosters |>
  select(season, team, headshot_url, gsis_it_id) |>
  mutate(id = as.integer(gsis_it_id))

data <- read.csv('nice_undercutting_results.csv')

ranking_df <- data |>
  filter(is_undercutting == 'True') |>
  mutate(diff = predicted - actual) |>
  group_by(nfl_id) |>
  summarize(
    name = first(player_name),
    rating = mean(diff),
    n_plays = n(),
    position = first(player_position),
    epa = mean(expected_points_added),
    man_rate = mean(team_coverage_man_zone == 'MAN_COVERAGE', na.rm = TRUE)
  ) |>
  filter(n_plays >= 10) |>
  mutate(rating = round(rating, 3)) |>
  left_join(rost_info_needed, by = c('nfl_id' = 'id')) |>
  left_join(teams_colors_logos, by = c('team' = 'team_abbr'))

cornerbacks <- ranking_df |>
  filter(position == 'CB') |>
  slice_max(order_by = rating, n = 10) |>
  mutate(epa = round(epa, 2)) |>
  mutate(man_rate = round(man_rate, 2))

final_viz <- cornerbacks |>
  select(team_logo_espn, headshot_url, name, rating, n_plays, epa, man_rate) |>
  gt() |>
  
  gtExtras::gt_img_rows(team_logo_espn, height = 30) |>
  gtExtras::gt_img_rows(headshot_url, height = 35) |>
  
  cols_label(
    team_logo_espn = "Team",
    headshot_url   = "",
    name         = "Player",
    rating         = "JUDGE",
    n_plays        = "Routes Jumped",
    epa            = 'EPA / Play',
    man_rate       = 'Man Rate'
  ) |>
  
  cols_align(
    align = "center",
    columns = everything()
  ) |>
  
  gtExtras::gt_theme_dark() |>
  
  data_color(
    columns = rating,
    colors = scales::col_numeric(
      c("#2CB1BC", "#0B525B"),
      domain = range(cornerbacks$rating, na.rm = TRUE)
    )
  ) |>
  
  data_color(
    columns = n_plays,
    colors = scales::col_numeric(
      c("#8FAADC", "#2F4A7D"),
      domain = range(cornerbacks$n_plays, na.rm = TRUE)
    )
  ) |>
  
  data_color(
    columns = epa,
    colors = scales::col_numeric(
      c("#C9A33A", "#6F5200"),
      domain = range(cornerbacks$epa, na.rm = TRUE)
    )
  ) |>
  
  data_color(
    columns = man_rate,
    colors = scales::col_numeric(
      c("#D28C8C", "#7A3E3E"),
      domain = range(cornerbacks$man_rate, na.rm = TRUE)
    )
  ) |>
  
  tab_header(
    title    = md("**Cornerback Leaders in JUDGE, 2023**"),
    subtitle = md("JUDGE = Jump Underneath Distance Gained over Expected")
  ) |>
  
  tab_options(
    data_row.padding = px(6),
    table.border.top.color = "transparent",
    table.border.bottom.color = "transparent"
  ) |>
  
  opt_align_table_header(align = "center") |>
  tab_source_note(source_note = md("Matthew Cooper & Dayyan Hamid - NFL Big Data Bowl 2026 - Min. 10 Undercuts"))

gtsave(final_viz, 'JUDGE_leaders_2023.png')


##########################################################################################################


library(ggrepel)

cbs <- ranking_df |> filter(position == "CB")

cbs |>
  ggplot(aes(x = rating, y = epa)) +
  geom_hline(yintercept = mean(cbs$epa), linetype = "dashed") +
  geom_vline(xintercept = mean(cbs$rating), linetype = "dashed") +
  geom_image(aes(image = team_logo_espn), size = 0.05, asp = 16/9) +
  ggrepel::geom_text_repel(
    aes(label = name),
    nudge_y = 0.1,
    size = 4,
    fontface = "bold",
    max.overlaps = 20
  ) +
  theme_minimal() +
  labs(
    x = "JUDGE (Distance Gained Over Expected)",
    y = "EPA per Play",
    title = "Cornerback EPA vs JUDGE (2023)",
    subtitle = "Underneath routes only"
  ) +
  scale_x_continuous(breaks = scales::pretty_breaks(n = 8)) +
  scale_y_continuous(breaks = scales::pretty_breaks(n = 8))

x_bar <- mean(cbs$rating, na.rm = TRUE)
y_bar <- mean(cbs$epa, na.rm = TRUE)

x_rng <- diff(range(cbs$rating, na.rm = TRUE))
y_rng <- diff(range(cbs$epa, na.rm = TRUE))

cbs |>
  ggplot(aes(x = rating, y = epa)) +
  
  # Quadrant shading
  geom_rect(aes(xmin = -Inf, xmax = x_bar, ymin = -Inf, ymax = y_bar),
            fill = "#D6EAF8", alpha = 0.25, inherit.aes = FALSE) +
  geom_rect(aes(xmin = x_bar, xmax = Inf, ymin = -Inf, ymax = y_bar),
            fill = "#D5F5E3", alpha = 0.25, inherit.aes = FALSE) +
  geom_rect(aes(xmin = -Inf, xmax = x_bar, ymin = y_bar, ymax = Inf),
            fill = "#F2D7D5", alpha = 0.25, inherit.aes = FALSE) +
  geom_rect(aes(xmin = x_bar, xmax = Inf, ymin = y_bar, ymax = Inf),
            fill = "#FCF3CF", alpha = 0.25, inherit.aes = FALSE) +
  
  geom_hline(yintercept = y_bar, linetype = "dashed") +
  geom_vline(xintercept = x_bar, linetype = "dashed") +
  
  # Logos
  geom_image(aes(image = team_logo_espn),
             size = 0.04, asp = 16/9) +
  
  ggrepel::geom_label_repel(
    aes(label = name),
    size = 4,
    fontface = "bold",
    color = "black",
    fill = "white",
    label.size = 0.15,
    label.padding = 0.15,
    box.padding = 0.6,
    point.padding = 0.5,
    force = 2,
    max.overlaps = Inf
  ) +
  annotate("label",
           x = x_bar - 0.35 * x_rng, y = y_bar + 0.35 * y_rng,
           label = "Low Performers",
           fontface = "bold",
           size = 4,
           color = "white",
           fill = "#111111",
           label.size = 0,
           alpha = 0.9) +
  annotate("label",
           x = x_bar + 0.35 * x_rng, y = y_bar + 0.35 * y_rng,
           label = "Underperforming",
           fontface = "bold",
           size = 4,
           color = "white",
           fill = "#111111",
           label.size = 0,
           alpha = 0.9) +
  annotate("label",
           x = x_bar - 0.35 * x_rng, y = y_bar - 0.35 * y_rng,
           label = "Opportunistic",
           fontface = "bold",
           size = 4,
           color = "white",
           fill = "#111111",
           label.size = 0,
           alpha = 0.9) +
  annotate("label",
           x = x_bar + 0.35 * x_rng, y = y_bar - 0.35 * y_rng,
           label = "High Performers",
           fontface = "bold",
           size = 4,
           color = "white",
           fill = "#111111",
           label.size = 0,
           alpha = 0.9) +
  # Layout fixes
  coord_cartesian(clip = "off") +
  theme_minimal() +
  theme(
    plot.title = element_text(
      hjust = 0.5,
      face = "bold",
      size = 18
    ),
    plot.subtitle = element_text(
      hjust = 0.5,
      face = "bold",
      size = 12,
      color = "gray40"
    ),
    plot.caption = element_text(
      hjust = 1,
      size = 9,
      color = "gray50"
    )
  ) +
  labs(
    x = "JUDGE (Distance Gained Over Expected)",
    y = "EPA / Target",
    title = "Cornerback EPA Allowed vs JUDGE (2023)",
    subtitle = "Underneath routes only - Min. 10 Undercuts",
    caption = "Matthew Cooper & Dayyan Hamid — NFL Big Data Bowl 2026"
  ) +
  scale_x_continuous(
    limits = range(cbs$rating, na.rm = TRUE),
    breaks = scales::pretty_breaks(8)
  ) +
  scale_y_continuous(breaks = scales::pretty_breaks(8))

ggsave(
  "cornerback_epa_vs_judge_2023.png",
  width = 14,
  height = 8,
  dpi = 400,
  bg = "white"
)
