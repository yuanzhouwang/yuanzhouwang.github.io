# app.R

## --- AI-Generated fix for web integration --- ##

# When running under Shinylive in the browser, install needed CRAN packages
if (Sys.getenv("IN_SHINYLIVE") == "1" && requireNamespace("webr", quietly = TRUE)) {
  # Either use webr::install(...)
  webr::install(c("shiny", "ggplot2", "viridisLite", "scales"))
  # Or, if you prefer, shim install.packages() and use that:
  # webr::shim_install()
  # install.packages(c("shiny", "ggplot2", "viridisLite", "scales"), quiet = TRUE)
}

## -------------------------------------------- ##

library(shiny)
library(ggplot2)

ui <- fluidPage(
  titlePanel("Q(P) model with g(Q), ρ(Q), and f(B)"),
  sidebarLayout(
    sidebarPanel(
      numericInput("N", "N (population cap)", value = 50, min = 1, step = 1),
      numericInput("T", "T (threshold)", value = 50, min = 0, step = 1),
      numericInput("d", "d (baseline)", value = 0, step = 0.1),
      
      radioButtons("rhoType", "ρ(Q) form",
                   c("Olsonian: ρ(Q) = Q / g(Q)" = "olsonian",
                     "Lumpy: ρ(Q) = (T - Q) / g(Q)" = "lumpy")),
      
      radioButtons("gType", "g(Q) form (auto-normalized)",
                   c("Logistic" = "logistic",
                     "Exponential" = "exponential",
                     "Linear" = "linear",
                     "Superlinear" = "superlinear")),
      
      numericInput("eps", "Calibrate g(Q_anchor) = 1 - ε  (ε)",
                  min = 1e-4, max = 0.1, value = 0.01, step = 1e-4),
      
      conditionalPanel(
        condition = "input.gType == 'logistic'",
        numericInput("g0", "Set g(0) = g0 (logistic only)",
                    min = 1e-6, max = 0.49, value = 0.05, step = 1e-3)
      ),
      
      # g0 also for Linear/Superlinear
      conditionalPanel(
        condition = "input.gType == 'linear' || input.gType == 'superlinear'",
        numericInput("g0_ls", "Set g(0) = g0 (linear/superlinear)",
                    min = 1e-6, max = 0.49, value = 0.05, step = 1e-3)
      ),
      
      # exponent for Superlinear (p > 1)
      conditionalPanel(
        condition = "input.gType == 'superlinear'",
        numericInput("p_super", "Superlinear power p (>1)",
                    min = 1.1, max = 5, value = 2, step = 0.1)
      ),
      
      radioButtons("fType", "f(B) distribution",
                   c("Normal(μ, σ)" = "normal",
                     "Uniform[a, b]" = "uniform",
                     "Polarized" = "polarized",
                     "Superpolar" = "superpol")),
      
      conditionalPanel(
        condition = "input.fType == 'normal'",
        numericInput("mu", "μ (mean)", value = 12, step = 0.1),
        numericInput("sd", "σ (sd)", value = 30, min = 1e-6, step = 0.1)
      ),
      conditionalPanel(
        condition = "input.fType == 'uniform'",
        numericInput("umin", "a (min)", value = 0, step = 0.1),
        numericInput("umax", "b (max)", value = 100, step = 0.1)
      ),
      conditionalPanel(
        condition = "input.fType == 'polarized'",
        numericInput("w_mix", "Weight on component 1 (w)", min = 0.01, max = 0.99, value = 0.5, step = 0.01),
        numericInput("mu1", "μ1 (mean 1)", value = -20, step = 0.1),
        numericInput("sd1", "σ1 (sd 1)",  value = 12,  min = 1e-6, step = 0.1),
        numericInput("mu2", "μ2 (mean 2)", value = 20,  step = 0.1),
        numericInput("sd2", "σ2 (sd 2)",  value = 12,  min = 1e-6, step = 0.1)
      ),
      conditionalPanel(
        condition = "input.fType == 'superpol'",
        numericInput("q_a", "a (min)", value = 0, step = 0.1),
        numericInput("q_b", "b (max)", value = 10, step = 0.1),
        helpText("Quadratic shape: w(x) = α + β (x - m) + γ (x - m)^2, with m = (a+b)/2."),
        numericInput("q_alpha", "α (constant term)", value = 1, step = 0.1),
        numericInput("q_beta",  "β (linear term)",   value = 0, step = 0.1),
        numericInput("q_gamma", "γ (quadratic term)",value = 0.06, step = 0.1)
      ),
      
      hr(),
      h4("P range for Q(P)"),
      # Enforce P > 0
      numericInput("Pmin", "P min (> 0)", value = 0.45, min = 1e-6, step = 0.1),
      numericInput("Pmax", "P max", value = 0.58, min = 0.2, step = 0.5),
      sliderInput("nP", "Resolution (# P points)", min = 50, max = 800, value = 100, step = 10),
      
      hr(),
      helpText("Notes:",
               "• g is auto-calibrated so it is ~1 at Q=N (Olsonian) or Q=T (Lumpy).",
               "• Logistic also matches g(0)=g0.",
               "• Solver respects domain: Olsonian 0<Q<N; Lumpy 0<Q<T.")
    ),
    mainPanel(
      tabsetPanel(
        tabPanel("Q(P) & Backlash Diagnostic",
                 plotOutput("qp_plot", height = 300),
                 plotOutput("SP_plot", height = 300),
                 verbatimTextOutput("S_diag"),
                 verbatimTextOutput("param_snapshot")),
        tabPanel("g(Q), ρ′(Q), Elasticity",
                 plotOutput("g_plot", height = 260),
                 plotOutput("rhoPrime_plot", height = 260),
                 plotOutput("elasticity_plot", height = 260)),
        tabPanel("f(B) distribution",
                 plotOutput("f_plot", height = 360),
                 plotOutput("BQ_plot", height = 360)),
        tabPanel("Diagnostics",
                 sliderInput("P_check", "Choose P for diagnostics",
                             min = 0.1, max = 5, value = 1, step = 0.1),
                 verbatimTextOutput("point_diag"),     # equilibrium Q*, B*, residual
                 verbatimTextOutput("endpoint_checks") # existing near-boundary table
        )
      )
    )
  )
)

server <- function(input, output, session) {
  tiny <- 1e-12
  clip01 <- function(x) pmin(pmax(x, 0), 1)
  
  # Keep Pmax >= Pmin and P_check within range
  observe({
    if (input$Pmax <= input$Pmin) {
      updateNumericInput(session, "Pmax", value = input$Pmin + 0.1)
    }
    updateSliderInput(session, "P_check",
                      min = input$Pmin, max = input$Pmax,
                      value = min(max(input$P_check, input$Pmin), input$Pmax))
  })
  
  Q_anchor <- reactive({
    if (input$rhoType == "olsonian") max(input$N, tiny) else max(input$T, tiny)
  })
  
  gg <- reactive({
    Qa  <- Q_anchor()
    eps <- input$eps
    tiny <- 1e-12
    
    if (input$gType == "logistic") {
      g0 <- min(max(input$g0, 1e-12), 0.499999)
      num <- log((1 - g0) / g0) - log(eps / (1 - eps))
      k   <- max(num / Qa, tiny)
      q0  <- log((1 - g0) / g0) / k
      gfun  <- function(q) 1 / (1 + exp(-k * (q - q0)))
      gpfun <- function(q) { gq <- gfun(q); k * gq * (1 - gq) }
      return(list(type="logistic", g=gfun, gp=gpfun, k=k, q0=q0))
    }
    
    if (input$gType == "exponential") {
      k <- max(log(1/eps) / Qa, tiny)
      gfun  <- function(q) 1 - exp(-k * pmax(q, 0))
      gpfun <- function(q) k * exp(-k * pmax(q, 0))
      return(list(type="exponential", g=gfun, gp=gpfun, k=k, q0=0))
    }
    
    # ----- Linear & Superlinear (calibrated to g(Qa) = 1 - eps) -----
    g0_ls <- min(max(input$g0_ls, 1e-12), 1 - eps - 1e-12)  # keep < 1 - eps
    A     <- (1 - eps) - g0_ls                              # amplitude to hit 1 - eps at Qa
    
    if (input$gType == "linear") {
      # g(q) = g0_ls + A * (q / Qa)
      gfun  <- function(q) g0_ls + A * (pmax(q, 0) / Qa)
      gpfun <- function(q) rep(A / Qa, length(q))
      return(list(type="linear", g=gfun, gp=gpfun, g0=g0_ls, A=A, Qa=Qa))
    }
    
    if (input$gType == "superlinear") {
      p <- max(input$p_super, 1 + 1e-6)  # ensure > 1
      # g(q) = g0_ls + A * (q / Qa)^p
      gfun  <- function(q) g0_ls + A * (pmax(q, 0) / Qa)^p
      gpfun <- function(q) {
        x <- pmax(q, 0) / Qa
        A * p * (x^(p - 1)) / Qa
      }
      return(list(type="superlinear", g=gfun, gp=gpfun, g0=g0_ls, A=A, p=p, Qa=Qa))
    }
  })
  
  rhoPieces <- reactive({
    g  <- gg()$g
    gp <- gg()$gp
    Tval <- input$T
    
    if (input$rhoType == "olsonian") {
      rho  <- function(q) {
        gq <- pmax(g(q), tiny)
        q / gq
      }
      rhop <- function(q) {
        gq  <- pmax(g(q), tiny)
        gpq <- gp(q)
        (gq - q * gpq) / (gq^2)
      }
      upper <- function() input$N
    } else {
      rho  <- function(q) {
        gq <- pmax(g(q), tiny)
        (Tval - q) / gq
      }
      rhop <- function(q) {
        gq  <- pmax(g(q), tiny)
        gpq <- gp(q)
        -(gq + (Tval - q) * gpq) / (gq^2)
      }
      upper <- function() input$T
    }
    list(rho=rho, rhop=rhop, upper=upper)
  })
  
  f_and_F <- reactive({
    clip01 <- function(x) pmin(pmax(x, 0), 1)
    finite <- function(x, fallback) { x <- suppressWarnings(as.numeric(x)); if (!is.finite(x)) fallback else x }
    
    if (input$fType == "normal") {
      mu <- finite(input$mu, 0)
      sd <- finite(input$sd, 1)
      if (sd <= 0) sd <- 1
      f <- function(x) dnorm(x, mean = mu, sd = sd)
      F <- function(x) pnorm(x, mean = mu, sd = sd)
      support <- function() c(mu - 4*sd, mu + 4*sd)
      return(list(pdf = f, cdf = F, support = support))
    }
    
    if (input$fType == "uniform") {
      a <- finite(input$umin, 0)
      b <- finite(input$umax, 1)
      if (b <= a) b <- a + 1e-6
      f <- function(x) ifelse(x >= a & x <= b, 1/(b-a), 0)
      F <- function(x) clip01((x - a) / (b - a))
      support <- function() c(a, b)
      return(list(pdf = f, cdf = F, support = support))
    }
    
    if (input$fType == "polarized") {
      w  <- min(max(finite(input$w_mix, 0.5), 1e-6), 1-1e-6)
      m1 <- finite(input$mu1, -1)
      s1 <- max(finite(input$sd1, 1), 1e-6)
      m2 <- finite(input$mu2,  1)
      s2 <- max(finite(input$sd2, 1), 1e-6)
      
      f <- function(x) w * dnorm(x, m1, s1) + (1 - w) * dnorm(x, m2, s2)
      F <- function(x) w * pnorm(x, m1, s1) + (1 - w) * pnorm(x, m2, s2)
      
      # cover both modes ± 4σ for each component
      lo <- min(m1 - 4*s1, m2 - 4*s2)
      hi <- max(m1 + 4*s1, m2 + 4*s2)
      support <- function() c(lo, hi)
      
      return(list(pdf = f, cdf = F, support = support))
    }
    
    if (input$fType == "superpol") {
      finite <- function(x, fb) { x <- suppressWarnings(as.numeric(x)); if (is.finite(x)) x else fb }
      
      a <- finite(input$q_a, 0)
      b <- finite(input$q_b, 1)
      if (!is.finite(a)) a <- 0
      if (!is.finite(b) || b <= a) b <- a + 1e-6
      
      alpha <- finite(input$q_alpha, 1)
      beta  <- finite(input$q_beta,  0)
      gamma <- finite(input$q_gamma, 0)
      
      # Center at midpoint for numerical stability
      m  <- 0.5 * (a + b)
      nx <- 2048L                      # dense grid for smooth CDF
      xg <- seq(a, b, length.out = nx)
      w  <- alpha + beta * (xg - m) + gamma * (xg - m)^2
      w  <- pmax(w, 0)                 # ensure nonnegative "weight"
      
      # If the quadratic is nonpositive everywhere, fall back to uniform on [a,b]
      if (all(w <= 0)) {
        pdf_vals <- rep(1 / (b - a), nx)
        cdf_vals <- (xg - a) / (b - a)
      } else {
        dx       <- (b - a) / (nx - 1)
        Z        <- sum(w) * dx
        if (!is.finite(Z) || Z <= 0) Z <- 1 # safety
        pdf_vals <- w / Z
        
        # cumulative via trapezoid rule, capped to [0,1]
        cdf_vals <- cumsum(c(0, head(pdf_vals, -1))) * dx
        # small correction to reach 1 exactly at the end
        cdf_vals <- pmin(pmax(cdf_vals, 0), 1)
        cdf_vals[length(cdf_vals)] <- 1
      }
      
      # Fast interpolators for pdf and cdf
      f_pdf <- approxfun(xg, pdf_vals, yleft = 0, yright = 0, ties = "ordered")
      F_cdf <- approxfun(xg, cdf_vals, yleft = 0, yright = 1, ties = "ordered")
      
      support <- function() c(a, b)
      return(list(pdf = f_pdf, cdf = F_cdf, support = support))
    }
  })
  
  # h(Q;P) residual
  h_fun <- reactive({
    N <- input$N; d <- input$d
    F <- f_and_F()$cdf
    rho <- rhoPieces()$rho
    function(q, P) {
      Bstar <- (P - d) * rho(q)
      q - N * (1 - F(Bstar))
    }
  })
  
  # Domain restricted solver: Olsonian (0,Q,N) ; Lumpy (0,Q,T)
  solve_Q_for_P <- function(P) {
    N <- input$N
    upper <- rhoPieces()$upper()
    epsQ <- min(1e-8 * max(N, upper, 1), 1e-6)
    
    lo <- epsQ
    hi <- max(upper - epsQ, lo + epsQ)  # ensure lo < hi
    h <- h_fun()
    
    # Try to bracket strictly inside (lo, hi)
    Qgrid <- seq(lo, hi, length.out = 601)
    hv <- sapply(Qgrid, function(q) h(q, P))
    idx <- which(diff(sign(hv)) != 0)
    
    if (length(idx) >= 1) {
      loB <- Qgrid[idx[1]]
      hiB <- Qgrid[idx[1] + 1]
      out <- try(uniroot(function(q) h(q, P), lower = loB, upper = hiB), silent = TRUE)
      if (!inherits(out, "try-error")) return(out$root)
    }
    
    # No sign change: clamp to nearest interior boundary point
    if (abs(h(lo, P)) < abs(h(hi, P))) lo else hi
  }
  
  # Find all Q roots of h(Q; P) = Q - N*(1 - F((P-d)*rho(Q))) for a given P
  all_Q_for_P <- function(P, n_scan = 400) {
    N <- input$N
    upperQ <- if (input$rhoType == "olsonian") input$N else input$T
    epsQ <- max(upperQ * 1e-8, 1e-10)
    
    rho <- rhoPieces()$rho
    F   <- f_and_F()$cdf
    
    h <- function(Q) Q - N * (1 - F((P - input$d) * rho(Q)))
    
    # coarse grid to find sign changes
    qg <- seq(epsQ, upperQ - epsQ, length.out = n_scan)
    hg <- sapply(qg, h)
    
    roots <- numeric(0)
    for (k in 1:(length(qg) - 1)) {
      y1 <- hg[k]; y2 <- hg[k + 1]
      if (!is.finite(y1) || !is.finite(y2)) next
      if (y1 == 0) { roots <- c(roots, qg[k]); next }
      if (y1 * y2 < 0) {
        # refine with uniroot in the bracket
        r <- try(uniroot(h, lower = qg[k], upper = qg[k + 1]), silent = TRUE)
        if (!inherits(r, "try-error")) roots <- c(roots, r$root)
      }
    }
    sort(unique(roots))
  }
  
  qp_data <- reactive({
    Pseq <- seq(max(input$Pmin, 1e-9), input$Pmax, length.out = input$nP)
    N <- input$N; d <- input$d
    rho  <- rhoPieces()$rho
    rhop <- rhoPieces()$rhop
    Fcdf <- f_and_F()$cdf
    fpdf <- f_and_F()$pdf
    
    Qs <- vapply(Pseq, solve_Q_for_P, numeric(1))
    Bstar <- (Pseq - d) * rho(Qs)
    Sp <- N * fpdf(Bstar) * (Pseq - d) * rhop(Qs)
    
    df <- data.frame(P = Pseq, Q = Qs, Bstar = Bstar, S = Sp)
    
    df$S_theory <- NA_real_
    
    if (input$rhoType == "olsonian" && input$gType == "superlinear" && input$fType == "uniform") {
      N  <- input$N
      a  <- input$umin; b <- input$umax
      g0 <- input$g0_ls
      eps <- input$eps
      p   <- input$p_super
      A   <- (1 - eps) - g0
      
      x <- pmin(pmax(df$Q / N, 1e-12), 1 - 1e-12)         # x = Q/N
      gQ <- g0 + A * (x^p)                                 # g(Q)
      # Closed-form S(x) for uniform[a,b] with superlinear g:
      # S(x) = (1 - x) * [ 1/x - (A p x^(p-1)) / (g0 + A x^p) ]
      df$S_theory <- (1 - x) * (1/x - (A * p * x^(p - 1)) / gQ)
      
      # Optional: record whether B* lies inside support (it should; else f(B*)=0)
      df$B_inside <- (df$Bstar >= a) & (df$Bstar <= b)
    }
    
    df
  })
  
  qp_multi <- reactive({
    Pseq <- seq(input$Pmin, input$Pmax, length.out = input$nP)
    rho <- rhoPieces()$rho
    rhop <- rhoPieces()$rhop
    pdf <- f_and_F()$pdf
    
    out <- lapply(Pseq, function(P) {
      Qs <- all_Q_for_P(P)
      if (length(Qs) == 0) return(NULL)
      B  <- (P - input$d) * rho(Qs)
      S  <- input$N * pdf(B) * (P - input$d) * rhop(Qs)
      data.frame(P = P, Q = Qs, Bstar = B, S = S)
    })
    do.call(rbind, out)
  })
  
  # Intervals of P where S(P) < -1, computed from qp_data()
  S_intervals <- reactive({
    df <- qp_data()
    valid <- which(!is.na(df$S))
    if (length(valid) == 0) return(data.frame(Plo = numeric(0), Phi = numeric(0)))
    
    idx <- which(df$S < -1 & !is.na(df$S))
    if (length(idx) == 0) return(data.frame(Plo = numeric(0), Phi = numeric(0)))
    
    # group contiguous indices into runs
    runs <- split(idx, cumsum(c(1, diff(idx) != 1)))
    out <- do.call(rbind, lapply(runs, function(idxs) {
      data.frame(Plo = min(df$P[idxs]), Phi = max(df$P[idxs]))
    }))
    rownames(out) <- NULL
    out
  })
  
  # ----- Plots / Outputs -----
  
  # ---- Q(P) plot with multi-root overlay (no warnings) ----
  output$qp_plot <- renderPlot({
    df1 <- qp_data()      # your original single-branch curve
    dfm <- qp_multi()     # all equilibria across P (possibly 0/1/2+ per P)
    
    ggplot() +
      # gray points: all equilibria at each P (no "group" needed)
      { if (!is.null(dfm) && nrow(dfm) > 0)
        geom_point(data = dfm, aes(P, Q),
                   color = "grey50", alpha = 0.7, size = 0.9) } +
      # tracked branch as a solid black line
      { if (!is.null(df1) && nrow(df1) > 1)
        geom_line(data = df1, aes(P, Q), color = "black", linewidth = 1.1) } +
      labs(title = "Equilibria Q(P): black = tracked branch, grey = all roots",
           x = "P", y = "Q") +
      theme_minimal(base_size = 13)
  })
  
  output$SP_plot <- renderPlot({
    df1 <- qp_data()    # tracked branch: columns include P, S
    dfm <- qp_multi()   # all equilibria across P: columns include P, S
    
    # Min across branches at each P (envelope)
    agg <- NULL
    if (!is.null(dfm) && nrow(dfm) > 0) {
      agg <- aggregate(S ~ P, data = dfm, FUN = min)
      agg <- agg[order(agg$P), , drop = FALSE]
    }
    
    ggplot() +
      # All equilibria (possibly 1 or more per P): grey points
      { if (!is.null(dfm) && nrow(dfm) > 0)
        geom_point(data = dfm, aes(x = P, y = S),
                   color = "grey50", alpha = 0.7, size = 0.9) } +
      # Tracked branch from qp_data(): black line
      { if (!is.null(df1) && nrow(df1) > 1)
        geom_line(data = df1, aes(x = P, y = S),
                  color = "black", linewidth = 1.1) } +
      # Envelope: min S across branches at each P (red dashed)
      { if (!is.null(agg) && nrow(agg) > 1)
        geom_line(data = agg, aes(x = P, y = S),
                  color = "red", linewidth = 1, linetype = "dashed") } +
      # Backlash threshold
      geom_hline(yintercept = -1, linetype = "dashed", color = "red") +
      labs(title = "Backlash diagnostic: S(P) across multiple equilibria",
           subtitle = "Grey = all equilibria; Black = tracked branch; Red dashed = min across branches; S=-1 dashed",
           x = "P", y = "S(P)") +
      theme_minimal(base_size = 13)
  })
  
  output$S_diag <- renderPrint({
    cat("S(P) = N f(B*) (P - d) ρ′(Q)\n")
    dfm <- qp_multi()
    if (is.null(dfm) || nrow(dfm) == 0) {
      cat("No equilibria found in the current P-range.\n"); return(invisible())
    }
    # For each P, take min S across branches
    agg <- aggregate(S ~ P, data = dfm, FUN = min)
    mS  <- min(agg$S, na.rm = TRUE); MS <- max(agg$S, na.rm = TRUE)
    cat(sprintf("minS (across branches): %.4f   maxS: %.4f\n", mS, MS))
    cat("Backlash requires S < -1\n")
    
    idx <- which(agg$S < -1)
    if (length(idx) == 0) {
      cat("\nNo P-intervals where S < -1 across branches.\n")
    } else {
      runs <- split(idx, cumsum(c(1, diff(idx) != 1)))
      cat("\nP-intervals where min_branch S(P) < -1:\n")
      for (r in runs) {
        cat(sprintf("  [%.6f, %.6f]\n", min(agg$P[r]), max(agg$P[r])))
      }
    }
  })
  
  output$g_plot <- renderPlot({
    g <- gg()$g
    Qmax <- max(input$N, input$T)
    q <- seq(0, Qmax, length.out = 400)
    df <- data.frame(Q=q, g = g(q))
    ggplot(df, aes(Q, g)) +
      geom_line(linewidth = 1) +
      labs(title = "g(Q)", y = "g(Q)", x = "Q") +
      theme_minimal(base_size = 13)
  })
  
  output$rhoPrime_plot <- renderPlot({
    rhop <- rhoPieces()$rhop
    Qmax <- max(input$N, input$T)
    q <- seq(0, Qmax, length.out = 400)
    df <- data.frame(Q=q, rhop = rhop(q))
    ggplot(df, aes(Q, rhop)) +
      geom_hline(yintercept = 0, linewidth = 0.4, linetype = "dashed") +
      geom_line(linewidth = 1) +
      labs(title = "ρ′(Q)", y = "ρ′(Q)", x = "Q") +
      theme_minimal(base_size = 13)
  })
  
  output$elasticity_plot <- renderPlot({
    g <- gg()$g
    gp <- gg()$gp
    Qmax <- max(input$N, input$T)
    q <- seq(0, Qmax, length.out = 400)
    gq <- pmax(g(q), tiny)
    eq <- (q * gp(q)) / gq
    df <- data.frame(Q=q, E=eq)
    ggplot(df, aes(Q, E)) +
      geom_hline(yintercept = 0, linewidth = 0.4, linetype = "dashed") +
      geom_line(linewidth = 1) +
      labs(title = "Elasticity:  E(Q) = Q g′(Q) / g(Q)", y = "E(Q)", x = "Q") +
      theme_minimal(base_size = 13)
  })
  
  output$f_plot <- renderPlot({
    distr <- f_and_F()
    rng <- distr$support()
    x <- seq(rng[1], rng[2], length.out = 400)
    y <- distr$pdf(x)
    ggplot(data.frame(x, y), aes(x, y)) +
      geom_line(linewidth = 1) +
      labs(title = "f(B): PDF", x = "B", y = "f(B)") +
      theme_minimal(base_size = 13)
  })
  
  output$BQ_plot <- renderPlot({
    # P, Q grids
    Pvals <- seq(input$Pmin, input$Pmax, length.out = 120)
    
    upperQ <- if (input$rhoType == "olsonian") input$N else input$T
    epsQ   <- max(upperQ * 1e-6, 1e-6)
    Qvals  <- seq(epsQ, upperQ - epsQ, length.out = 120)
    
    rho_fun <- rhoPieces()$rho
    d <- input$d
    
    grid <- expand.grid(P = Pvals, Q = Qvals)
    grid$Bstar <- (grid$P - d) * rho_fun(grid$Q)
    
    # winsorize for color scale so extremes don't wash out the palette
    finite_vals <- grid$Bstar[is.finite(grid$Bstar)]
    qlo <- if (length(finite_vals)) quantile(finite_vals, 0.01, na.rm = TRUE) else 0
    qhi <- if (length(finite_vals)) quantile(finite_vals, 0.99, na.rm = TRUE) else 1
    grid$Bplot <- pmin(pmax(grid$Bstar, qlo), qhi)
    
    # Equilibrium locus
    df_eq <- qp_data()[, c("P", "Q")]
    df_eq <- df_eq[is.finite(df_eq$P) & is.finite(df_eq$Q), , drop = FALSE]
    df_eq <- df_eq[order(df_eq$P), , drop = FALSE]
    
    # Diagnostic point (make a 1-row data frame so it doesn't inherit the grid)
    Q_star <- solve_Q_for_P(input$P_check)
    df_pt <- if (is.finite(Q_star)) data.frame(P = input$P_check, Q = Q_star) else NULL
    
    ggplot() +
      # Heatmap
      geom_raster(data = grid, aes(x = P, y = Q, fill = Bplot), interpolate = TRUE) +
      # Contours (use true B*, not winsorized), prevent aes inheritance
      geom_contour(
        data = grid,
        aes(x = P, y = Q, z = Bstar),
        color = "white", linewidth = 0.3, alpha = 0.8,
        inherit.aes = FALSE
      ) +
      # Equilibrium locus
      { if (nrow(df_eq) > 1)
        geom_path(data = df_eq, aes(x = P, y = Q), color = "red", linewidth = 1.1,
                  inherit.aes = FALSE) } +
      # Diagnostic point
      { if (!is.null(df_pt))
        geom_point(data = df_pt, aes(x = P, y = Q), color = "red", size = 2,
                   inherit.aes = FALSE) } +
      scale_fill_viridis_c(
        name = "B* = (P - d)·ρ(Q)",
        limits = c(qlo, qhi)
      ) +
      labs(title = "B*(Q,P) heatmap with equilibrium locus",
           x = "P", y = "Q") +
      theme_minimal(base_size = 13)
  })
  
  # ----- Point diagnostics: equilibrium values at P_check -----
  output$point_diag <- renderPrint({
    P  <- input$P_check
    d  <- input$d
    N  <- input$N
    rho <- rhoPieces()$rho
    F   <- f_and_F()$cdf
    
    Q_star <- solve_Q_for_P(P)
    B_star <- (P - d) * rho(Q_star)
    
    # Residual h(Q*; P) = Q* - N*(1 - F(B*))
    h_val  <- Q_star - N * (1 - F(B_star))
    
    cat("Equilibrium (interior) at selected P:\n")
    cat(sprintf("  P = %.6g\n", P))
    cat(sprintf("  Q* = %.6g\n", Q_star))
    cat(sprintf("  B* = (P - d) * ρ(Q*) = %.6g\n", B_star))
    cat(sprintf("  Residual h(Q*;P) = Q* - N*(1 - F(B*)) = %.3e\n", h_val))
  })
  
  # ----- Equilibrium endpoint diagnostics -----
  
  endpoint_diag <- reactive({
    P <- input$P_check
    N <- input$N; d <- input$d
    rho <- rhoPieces()$rho
    upper <- rhoPieces()$upper()
    F <- f_and_F()$cdf
    
    epsQ <- min(1e-8 * max(N, upper, 1), 1e-6)
    # Near-boundary interior points
    q_lo <- epsQ
    q_up <- upper - epsQ
    
    # Also report near T if Lumpy (since ρ(T)=0)
    q_T  <- if (input$rhoType == "lumpy") max(min(input$T - epsQ, q_up), q_lo) else NA_real_
    
    h <- h_fun()
    
    rows <- list(
      list(Point = "Near lower bound",  Q = q_lo, h = h(q_lo, P),
           Bstar = (P - d) * rho(q_lo)),
      list(Point = "Near upper bound",  Q = q_up, h = h(q_up, P),
           Bstar = (P - d) * rho(q_up))
    )
    if (input$rhoType == "lumpy") {
      rows <- append(rows, list(
        list(Point = "Near T (Lumpy)", Q = q_T, h = h(q_T, P),
             Bstar = (P - d) * rho(q_T))
      ))
    }
    
    df <- do.call(rbind, lapply(rows, as.data.frame))
    df$one_minus_F <- 1 - F(df$Bstar)
    df$RHS <- N * df$one_minus_F
    df$Equilibrium_if_allowed <- ifelse(abs(df$h) < 1e-6, "Yes (≈0)", "No")
    df
  })
  
  # --- Theoretical checker for: rho=Olsonian, g=Superlinear, f=Uniform[a,b] ---
  theory_band <- reactive({
    if (!(input$rhoType == "olsonian" && input$gType == "superlinear" && input$fType == "uniform"))
      return(NULL)
    
    N  <- input$N
    a  <- input$umin; b <- input$umax
    g0 <- input$g0_ls
    eps <- input$eps
    p   <- input$p_super
    if (!is.finite(N) || !is.finite(a) || !is.finite(b) || !is.finite(g0) ||
        !is.finite(eps) || !is.finite(p) || b <= a || g0 <= 0 || p <= 1) return(NULL)
    
    A <- (1 - eps) - g0
    if (A <= 0) return(NULL)
    
    # x := Q/N
    x <- seq(1e-6, 1 - 1e-6, length.out = 5000)
    gQ <- g0 + A * x^p                         # g(Q) with Q/N = x
    Sx <- (1 - x) * ( 1/x - (A * p * x^(p-1)) / gQ )
    
    # Uniform[a,b]: B* = a + (1 - x)(b - a), so P = d + (B* g(Q))/Q = d + (a + (1 - x)(b - a)) * gQ / (x * N)
    P_of_x <- input$d + (a + (1 - x) * (b - a)) * gQ / (x * N)
    
    # find S(x) < -1 segments and map to P
    idx <- which(Sx < -1)
    if (length(idx) == 0) {
      return(list(has_band = FALSE, minS = min(Sx, na.rm = TRUE)))
    }
    runs <- split(idx, cumsum(c(1, diff(idx) != 1)))
    band <- do.call(rbind, lapply(runs, function(idxs) {
      data.frame(Plo = min(P_of_x[idxs], na.rm = TRUE), Phi = max(P_of_x[idxs], na.rm = TRUE))
    }))
    list(
      has_band = TRUE,
      minS = min(Sx, na.rm = TRUE),
      band = band,
      P_at_min = P_of_x[which.min(Sx)],
      x_at_min = x[which.min(Sx)]
    )
  })
  
  numeric_band <- reactive({
    df <- qp_data()
    if (!all(c("P","S") %in% names(df))) return(NULL)
    idx <- which(df$S < -1 & is.finite(df$S))
    if (length(idx) == 0) return(list(has_band = FALSE, minS = min(df$S, na.rm = TRUE)))
    runs <- split(idx, cumsum(c(1, diff(idx) != 1)))
    band <- do.call(rbind, lapply(runs, function(idxs) {
      data.frame(Plo = min(df$P[idxs], na.rm = TRUE), Phi = max(df$P[idxs], na.rm = TRUE))
    }))
    list(has_band = TRUE, minS = min(df$S, na.rm = TRUE), band = band)
  })
  
  output$param_snapshot <- renderPrint({
    cat("Snapshot of current parameters (for sanity):\n")
    cat(sprintf("  rhoType=%s | gType=%s | fType=%s\n",
                input$rhoType, input$gType, input$fType))
    cat(sprintf("  N=%g  d=%g\n", input$N, input$d))
    if (input$gType == "superlinear") {
      cat(sprintf("  g0_ls=%g  p_super=%g  eps=%g  (so g(N)=1-eps)\n",
                  input$g0_ls, input$p_super, input$eps))
    }
    if (input$fType == "uniform") {
      cat(sprintf("  Uniform [a,b]=[%g, %g]\n", input$umin, input$umax))
    }
    cat(sprintf("  P-range = [%.6f, %.6f]  (resolution nP=%d)\n",
                input$Pmin, input$Pmax, input$nP))
  })
  
  output$endpoint_checks <- renderPrint({
    df <- endpoint_diag()
    cat("Residual h(Q;P) = Q - N * (1 - F((P-d)ρ(Q))) at interior near-boundaries:\n\n")
    print(within(df, {
      h <- round(h, 6); Q <- signif(Q, 6); Bstar <- signif(Bstar, 6); RHS <- signif(RHS, 6)
      one_minus_F <- signif(one_minus_F, 6)
    }), row.names = FALSE)
    cat("\nInterpretation: a zero residual would mean a boundary equilibrium (if boundaries were allowed).",
        "\n(We enforce 0<Q<N for Olsonian, 0<Q<T for Lumpy.)\n")
  })
}

shinyApp(ui, server)
