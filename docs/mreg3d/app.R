# app.R
# install.packages(c("shiny", "rgl", "rglwidget", "palmerpenguins", "dplyr"))

library(shiny)
library(rgl)
library(palmerpenguins)
library(dplyr)

ui <- fluidPage(
  titlePanel("3D Regression Visualization"),
  fluidRow(
    column(
      width = 9,
      rglwidgetOutput("rglPlot", width = "100%", height = "1000px")
    ),
    column(
      width = 3,
      h4("Data"),
      fileInput("file", "Upload a CSV (with headers)", accept = c(".csv")),
      helpText("If no file is uploaded, the app uses penguins (y=flipper_length_mm, x1=body_mass_g, x2=bill_length_mm)."),
      uiOutput("mapping_ui"),
      tags$hr(),
      h4("Change view"),
      actionButton("view_x1y",  "View x1–y plane"),
      br(), br(),
      actionButton("view_x2y",  "View x2–y plane"),
      br(), br(),
      actionButton("view_x1x2", "View x1–x2 plane")
    )
  )
)

server <- function(input, output, session) {
  
  #----- Helpers: make a default dataset (penguins) and a reader for uploads -----
  default_df <- reactive({
    penguins %>%
      select(
        body_mass_g,          # x1 default
        bill_length_mm,       # x2 default
        flipper_length_mm     # y default
      ) %>%
      rename(
        x1_default = body_mass_g,
        x2_default = bill_length_mm,
        y_default  = flipper_length_mm
      ) %>%
      na.omit()
  })
  
  uploaded_df <- reactive({
    req(input$file)
    # basic read; you can extend with sep/quote options if needed
    read.csv(input$file$datapath, check.names = FALSE)
  })
  
  # Current working data frame (uploaded if present, else default penguins)
  working_df <- reactive({
    if (!is.null(input$file)) {
      df <- uploaded_df()
    } else {
      df <- default_df() %>% rename(
        `body_mass_g` = x1_default,
        `bill_length_mm` = x2_default,
        `flipper_length_mm` = y_default
      )
    }
    validate(need(ncol(df) >= 3, "Need at least 3 columns."))
    df
  })
  
  #----- Column mapping UI (reactive to data) -----
  output$mapping_ui <- renderUI({
    df <- working_df()
    cols <- names(df)
    
    # sensible defaults if using penguins; otherwise just pick first 3
    default_y  <- if ("flipper_length_mm" %in% cols) "flipper_length_mm" else cols[min(3, length(cols))]
    default_x1 <- if ("body_mass_g"        %in% cols) "body_mass_g"        else cols[1]
    default_x2 <- if ("bill_length_mm"     %in% cols) "bill_length_mm"     else cols[min(2, length(cols))]
    
    tagList(
      selectInput("col_y",  "y (response)", choices = cols, selected = default_y),
      selectInput("col_x1", "x1 (predictor 1)", choices = cols, selected = default_x1),
      selectInput("col_x2", "x2 (predictor 2)", choices = cols, selected = default_x2)
    )
  })
  
  #----- Extract mapped vectors; validate numeric -----
  mapped_data <- reactive({
    df <- working_df()
    req(input$col_y, input$col_x1, input$col_x2)
    
    y_name  <- input$col_y
    x1_name <- input$col_x1
    x2_name <- input$col_x2
    
    validate(
      need(y_name  %in% names(df), "Chosen y column not found."),
      need(x1_name %in% names(df), "Chosen x1 column not found."),
      need(x2_name %in% names(df), "Chosen x2 column not found.")
    )
    
    # Pull columns
    y  <- df[[y_name]]
    x1 <- df[[x1_name]]
    x2 <- df[[x2_name]]
    
    # Try to coerce non-numeric to numeric (e.g., character numbers)
    if (!is.numeric(y))  y  <- suppressWarnings(as.numeric(y))
    if (!is.numeric(x1)) x1 <- suppressWarnings(as.numeric(x1))
    if (!is.numeric(x2)) x2 <- suppressWarnings(as.numeric(x2))
    
    # Drop rows with NAs after coercion
    keep <- complete.cases(y, x1, x2)
    y  <- y[keep]; x1 <- x1[keep]; x2 <- x2[keep]
    
    validate(
      need(length(y)  > 2, "Not enough valid rows after cleaning."),
      need(is.numeric(y)  && is.numeric(x1) && is.numeric(x2), "y, x1, and x2 must be numeric.")
    )
    
    list(
      y = y, x1 = x1, x2 = x2,
      labels = list(
        x1 = x1_name,
        x2 = x2_name,
        y  = y_name
      )
    )
  })
  
  #----- Drawing function (uses current mapping) -----
  draw_scene <- function(view = "default") {
    md <- mapped_data()
    x1 <- md$x1; x2 <- md$x2; y <- md$y
    labs <- md$labels
    
    # fits
    fit_plane <- lm(y ~ x1 + x2)
    fit_y_x1  <- lm(y ~ x1)
    fit_y_x2  <- lm(y ~ x2)
    
    b0 <- coef(fit_plane)["(Intercept)"]
    b1 <- coef(fit_plane)["x1"]
    b2 <- coef(fit_plane)["x2"]
    
    c0 <- coef(fit_y_x1)["(Intercept)"]
    c1 <- coef(fit_y_x1)["x1"]
    
    d0 <- coef(fit_y_x2)["(Intercept)"]
    d1 <- coef(fit_y_x2)["x2"]
    
    x1_seq <- seq(min(x1), max(x1), length.out = 50)
    x2_seq <- seq(min(x2), max(x2), length.out = 50)
    
    y_hat_x1_partial <- b0 + b1 * x1_seq + b2 * min(x2)
    y_hat_x1_simple  <- c0 + c1 * x1_seq
    y_hat_x2_partial <- b0 + b1 * min(x1) + b2 * x2_seq
    y_hat_x2_simple  <- d0 + d1 * x2_seq
    
    # Open an offscreen RGL device for shiny
    open3d(useNULL = TRUE)
    par3d(fov = 0)
    
    plot3d(
      x = x1, y = x2, z = y,
      xlab = paste0("x1 (", labs$x1, ")"),
      ylab = paste0("x2 (", labs$x2, ")"),
      zlab = paste0("y (",  labs$y,  ")"),
      size = 5, col = "black"
    )
    
    planes3d(-b1, -b2, 1, -b0, alpha = 0.4, col = "lightblue")
    
    # x1 lines
    lines3d(x1_seq, rep(min(x2), 50), y_hat_x1_partial, lwd = 4, col = "red")
    lines3d(x1_seq, rep(min(x2), 50), y_hat_x1_simple,  lwd = 2, col = "darkred")
    
    # x2 lines
    lines3d(rep(min(x1), 50), x2_seq, y_hat_x2_partial, lwd = 4, col = "blue")
    lines3d(rep(min(x1), 50), x2_seq, y_hat_x2_simple,  lwd = 2, col = "navy")
    
    # Labels
    text3d(
      x = min(x1), y = min(x2), z = max(y),
      texts = c(
        "Light Blue: Regression Model",
        "red: effect of x1 holding x2 fixed",
        "dark red: simple y ~ x1",
        "blue: effect of x2 holding x1 fixed",
        "navy: simple y ~ x2"
      ),
      adj = c(0,1), cex = 0.8, col = "black"
    )
    
    # Adjust view
    if (view == "x1y") {
      view3d(theta = 90, phi = 0, fov = 0)
    } else if (view == "x2y") {
      view3d(theta = 0, phi = 0, fov = 0)
    } else if (view == "x1x2") {
      view3d(theta = 0, phi = 90, fov = 0)
    } else {
      view3d(theta = 35, phi = 20, fov = 0)
    }
    
    rglwidget()
  }
  
  # Default view
  output$rglPlot <- renderRglwidget({
    draw_scene("default")
  })
  
  # Buttons to re-render with different views
  observeEvent(input$view_x1y, {
    output$rglPlot <- renderRglwidget({ draw_scene("x1y") })
  })
  observeEvent(input$view_x2y, {
    output$rglPlot <- renderRglwidget({ draw_scene("x2y") })
  })
  observeEvent(input$view_x1x2, {
    output$rglPlot <- renderRglwidget({ draw_scene("x1x2") })
  })
}

shinyApp(ui, server)
