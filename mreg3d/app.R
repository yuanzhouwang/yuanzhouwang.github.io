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
      rglwidgetOutput("rglPlot", width = "100%", height = "1200px")
    ),
    column(
      width = 3,
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
  
  # Function to draw the scene given a view
  draw_scene <- function(view = "default") {
    penguins_clean <- penguins %>%
      select(body_mass_g, bill_length_mm, flipper_length_mm) %>%
      na.omit()
    
    x1 <- penguins_clean$body_mass_g
    x2 <- penguins_clean$bill_length_mm
    y  <- penguins_clean$flipper_length_mm
    
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
    
    # Open scene
    open3d()
    par3d(fov = 0)
    
    plot3d(
      x = x1, y = x2, z = y,
      xlab = "x1 (body mass)",
      ylab = "x2 (bill length)",
      zlab = "y (flipper length)",
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
    
    # Adjust view depending on button
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
  
  # Each button re-renders with a new view
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