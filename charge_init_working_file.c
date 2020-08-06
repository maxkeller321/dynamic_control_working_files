/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * <h2><center>&copy; Copyright (c) 2020 STMicroelectronics.
  * All rights reserved.</center></h2>
  *
  * This software component is licensed by ST under Ultimate Liberty license
  * SLA0044, the "License"; You may not use this file except in compliance with
  * the License. You may obtain a copy of the License at:
  *                             www.st.com/SLA0044
  *
  ******************************************************************************
  */
/* USER CODE END Header */

/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "dwt_delay.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#define true 1
#define false 0
#define adc_maximum = 255;

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
TIM_HandleTypeDef htim8;

/* USER CODE BEGIN PV */

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
/* USER CODE BEGIN PFP */
void timer_8_pulse_counter_gpioc6_Init(void);
static void GPIO_for_AWG_Init(void);
void start_new_AWG_pulse_sequence(void);
void single_dynamic_output_pin_Init(uint16_t PIN);
int pow_int(int basis, int exponent);
void fill_the_AWG_input_bits(void);
void reset_awg_output_pins(void);
void timer1_Init(void);
void trigger_awg_2g(void);
void trigger_awg_128m(void);
void orange_input_pin_init(void);
void timer_2_Init(void);
void delay_100ns(uint32_t number_of_100ns);
void trigger_awg_Init(void);
void fill_the_AWG_128m_input_bits(void);

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
/* USER CODE BEGIN EXTI0_IRQn 0 */
uint8_t threshold = 3;
uint16_t Bit_12_0_for_AWG = 0x00;
uint8_t Bit_13_18_for_AWG = 0x00;


int AWG_sequence_id_1_2g = 1;
int AWG_sequence_id_2_2g = 4;

uint8_t AWG_sequence_id_1_128m = 1;
uint8_t AWG_sequence_id_2_128m = 3;

uint32_t bits_for_AWG_128m_1;
uint32_t bits_for_AWG_128m_2;

/**
  * @brief This function handles EXTI line0 interrupt.
  */

void EXTI0_IRQHandler(void)
{
	/* USER CODE BEGIN EXTI0_IRQn 0 */
	DWT_Delay(4);// important Delay

	GPIOA->ODR |= GPIO_PIN_1;
	GPIOA->ODR = 0x00;
	/* USER CODE END EXTI0_IRQn 0 */

	HAL_GPIO_EXTI_IRQHandler(GPIO_PIN_0);
	/* USER CODE BEGIN EXTI0_IRQn 1 */
  /* USER CODE END EXTI0_IRQn 1 */
}

/*
 AdcValue = ADC1->DR; // get the adc value
	      if (AdcValue >= ADC_UPGOING_LEVEL) {
	        current_level = 1;
	        count++;
	      }
	  }
	  else {
	        current_level = 0;
	      }
	   }
 */




uint8_t counting_time_ms = 3; // [ms]
uint32_t clock_frequency = 168000000; // Hz
uint32_t one_micro_second_while = 168000000/168000000; // while loop: while(time--) takes 4 clock cylces --> 1mu delay with while loop
uint32_t one_nano_second_while =  168000000/8000000;
/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{
  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */



	HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */
  timer1_Init();
  timer_8_pulse_counter_gpioc6_Init();
  trigger_awg_Init();
  GPIO_for_AWG_Init();
  orange_input_pin_init();
  fill_the_AWG_input_bits();
  fill_the_AWG_128m_input_bits();
  DWT_Init();
  timer_2_Init();


  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  /* USER CODE BEGIN 2 */
  uint16_t load_bits = GPIO_PIN_7 + GPIO_PIN_14;
  uint16_t bits_for_awg_sequence_2 = Bit_12_0_for_AWG + bits_for_AWG_128m_2;
  uint16_t bits_for_awg_sequence_1 = 0x00; // 0x01 + 0x0100000000;
  uint16_t bits_2_128m = bits_for_AWG_128m_2;
  uint16_t bits_1_128m = bits_for_AWG_128m_1;
  uint16_t count_variable = 0;

  /* USER CODE END 2 */
  //bits_for_AWG_128m_2 = 0x500;
  /* Infinite loop */
  while (1)
	  {
	    if(GPIOA->IDR & 0x04)
	    {
	      TIM8->CNT  &= 0x00000000; // 1: resets TIM8
	          while(GPIOA->IDR & 0x04) {

	          }

				//TIM8->CNT  &= 0x00000000; // 1: resets TIM8
				//DWT_Delay(800);
	          count_variable = TIM8->CNT;
	          if (TIM8->CNT >= threshold)
				{
					//GPIOB->ODR |= GPIO_PIN_15; // Set the data select bit on "High"
					//GPIOB->ODR |= Bit_13_18_for_AWG;
					//GPIOB->ODR |= GPIO_PIN_14; // Set the load bit on "High"
					//GPIOB->ODR = 0x00; // Set the load bit on "High"
					GPIOB->ODR = bits_for_awg_sequence_2;
					//GPIOB->ODR = Bit_12_0_for_AWG;
					//GPIOB->ODR |= bits_for_AWG_128m_2;
					GPIOB->ODR |= load_bits; // Set the load bit on "High" "128m"
					//GPIOB->ODR |= GPIO_PIN_14; // Set the load bit on "High"
					//GPIOB->ODR |= GPIO_PIN_7;//orange_level = 0;
					DWT_Delay(2);
					GPIOA->ODR |= GPIO_PIN_1; //trigger awg 2g
					GPIOA->ODR = 0x00;
					DWT_Delay(20);

					//GPIOB->ODR = 0x01;
					//GPIOB->ODR |= 0x0100000000;
					GPIOB->ODR = bits_for_awg_sequence_1;

					GPIOB->ODR |= load_bits; // Set the load bit on "High" "128m"
					DWT_Delay(1);

					//GPIOB->ODR |= GPIO_PIN_7;//orange_level = 0;
					//GPIOB->ODR |= GPIO_PIN_14; // Set the load bit on "High"

					//GPIOB->ODR = bits_for_awg_sequence_1;
					//GPIOB->ODR |= load_bits; // Set the load bit on "High" "128m"					//orange_level = 0;
					GPIOB->ODR = 0;
					//orange_level = 0;
				}
				else
				{
					DWT_Delay(10);
					GPIOA->ODR |= GPIO_PIN_1;
					GPIOA->ODR = 0x00;
				}
		 }
	/* USER CODE BEGIN WHILE */
	/* USER CODE END WHILE */
  }
  /* USER CODE BEGIN 3 */
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage 
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);
  /** Initializes the CPU, AHB and APB busses clocks 
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = 8;
  RCC_OscInitStruct.PLL.PLLN = 168;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = 7;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }
  /** Initializes the CPU, AHB and APB busses clocks 
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV4;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV2;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_5) != HAL_OK)
  {
    Error_Handler();
  }
}


/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};

  /* GPIO Ports Clock Enable */

  /*Configure GPIO pin : PA0 */
  GPIO_InitStruct.Pin = GPIO_PIN_0;
  GPIO_InitStruct.Mode = GPIO_MODE_IT_FALLING;
  GPIO_InitStruct.Pull = GPIO_PULLDOWN;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  /* EXTI interrupt init*/
  HAL_NVIC_SetPriority(EXTI0_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(EXTI0_IRQn);

}

/* USER CODE BEGIN 4 */

void timer_8_pulse_counter_gpioc6_Init(void){
	// After the call of this function, timer8 starts to count rising edges > 2V on PC6
	// Get counts with TIM8->CNT

	RCC->AHB1ENR |= 0x04; // 1: IO port C clock enabled

	// APB1 peripheral reset register
	RCC->APB2ENR  |= RCC_APB2ENR_TIM8EN; // 1: enable TIM8

	// GPIO port mode register (GPIOx_MODER)
	GPIOC->MODER  |= 0x2000; // 10: Alternate function mode PC6 => AF mode
	GPIOC->AFR[0] |= 0x3000000; // Must refer to AF3 (alternate function for TIM8)
	GPIOC->PUPDR  |= 0x2000;  // Sets pull down resistor for PA1

	// CCMR!: capture/compare mode register 1
	TIM8->CCMR1 |= 0x01; //  CC1 channel is configured as input, IC1 is mapped on TI1

	TIM8->SMCR |= 0x0007; // Bits[2:0]    111: External Clock Mode 1 - Rising edges of the selected trigger clock the counter.
	TIM8->SMCR |= 0x0050; // Bits[6:4]   110: selected Trigger: Filtered Timer Input 1 (TI1FP1)
	// 0x0050
	TIM8->ARR = 0xFFFF; // Set the timer reset on the highest possible value
	TIM8->PSC = 0x01; // disable prescaler
	TIM8->CR1  |= 0x0001; // Enable Timer
}

/**
  * @brief AWG output set function
  *	Before the use of this function:
    --> AWG_sequence_id for the pump & readout sequence has to be correct
    --> fill_the_AWG_input_bits has to be executed
  * @retval None
  */
void start_new_AWG_pulse_sequence(void)
{
	  // Set Bit 5-0 from the dynamic control cable as Bit 18-13 in the AWG sequence register
	  // Just use this Code if the DynamicSelectWidth is 1 in the Hardware Settings file of the dynamically controlled AWG
	  /*
	  GPIOB->ODR |= GPIO_PIN_15; // Set the data select bit on "High"
	  GPIOB->ODR |= Bit_13_18_for_AWG;
	  GPIOB->ODR |= GPIO_PIN_14; // Set the load bit on "High"
	  reset_awg_output_pins();
	  */

	  // Set Bit 12-0 from the dynamic control cable as Bit 12-0 in the AWG sequence register
	  // data select bit is already 0 (Pin 15)
	  GPIOB->ODR |= Bit_12_0_for_AWG;
	  GPIOB->ODR |= GPIO_PIN_14; // Set the load bit on "High"
	  reset_awg_output_pins();
}

/**
  * @brief AWG output reset function
  *	Sets all output voltages to "LOW"
  * @retval None
  */
void reset_awg_output_pins(void)
{
  GPIOB->ODR = 0x00000000;
}
/**
  * @brief AWG output pin init function
  *	initialize all GPIOB Pins in Output mode which are relevant for the AWG output
  * @retval None
  */
static void GPIO_for_AWG_Init(void)
{
  /* GPIOB Ports Clock Enable */
  __HAL_RCC_GPIOB_CLK_ENABLE();

  GPIOB->ODR = 0x00000000; // Sets all Output Pins on "Low"
  single_dynamic_output_pin_Init(GPIO_PIN_0);
  single_dynamic_output_pin_Init(GPIO_PIN_1);
  single_dynamic_output_pin_Init(GPIO_PIN_2);
  single_dynamic_output_pin_Init(GPIO_PIN_3);
  single_dynamic_output_pin_Init(GPIO_PIN_4);
  single_dynamic_output_pin_Init(GPIO_PIN_5);
  single_dynamic_output_pin_Init(GPIO_PIN_6);
  single_dynamic_output_pin_Init(GPIO_PIN_7);
  single_dynamic_output_pin_Init(GPIO_PIN_8);
  single_dynamic_output_pin_Init(GPIO_PIN_9);
  single_dynamic_output_pin_Init(GPIO_PIN_10);
  single_dynamic_output_pin_Init(GPIO_PIN_11);
  single_dynamic_output_pin_Init(GPIO_PIN_12);
  single_dynamic_output_pin_Init(GPIO_PIN_14);
  single_dynamic_output_pin_Init(GPIO_PIN_15);
}


void single_dynamic_output_pin_Init(uint16_t PIN)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};

  GPIO_InitStruct.Pin = PIN;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH; // Maybe GPIO_SPEED_FAST would be enough.
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);
}

void fill_the_AWG_128m_input_bits(void){
	uint8_t buffer_1 = AWG_sequence_id_1_128m;
	uint8_t buffer_2 = AWG_sequence_id_2_128m;
	for(int i = 0; i < 4; i++)
	{
		if (buffer_1 % 2 == 1)
			bits_for_AWG_128m_1 |= pow_int(2, i+8);
		buffer_1 /= 2;
		if (buffer_2 % 2 == 1)
			bits_for_AWG_128m_2 |= pow_int(2, i+8);
		buffer_2 /= 2;
	}
}


/**
  * Converts the global integer: AWG_sequence_id into the bitstrings:
  	  -->Bit_12_0_for_AWG & Bit_13_18_for_AWG
  	  The bitstrings are for the dynamical control of the AWG
  * @param None
  * @retval None
  */
void fill_the_AWG_input_bits(void)
{
	int buffer = AWG_sequence_id_2_2g;
	for(int i = 0; i < 13; i++)
	{
		if (buffer % 2 == 1)
			Bit_12_0_for_AWG |= pow_int(2, i);
		buffer /= 2;
	}
	for(int i = 0; i < 6; i++)
	{
		if (buffer % 2 == 1)
			Bit_13_18_for_AWG |= pow_int(2, i);
		buffer /= 2;
	}
}

int pow_int(int basis, int exponent){
	if (exponent == 0)
		return 1;
	else
		return basis*pow_int(basis, exponent-1);
}

/**
  * @brief TIM1 Initialization Function
  * Dont change the prescaler:
      it would also change the waiting time of: wait_100us();
  * @param None
  * @retval None
  */
void timer1_Init(void){
	RCC->APB2ENR  |= 0x01; // 1: enable TIM1 in clock register
	TIM1->CR1 |= 0x0001; //0001 Enable Timer
	TIM1->PSC = 1; // Prescaler lowers the timer frequency to a quarter
	TIM1->ARR = 0xFFFF;  // Set the timer reset on the highest possible value
}

void delay_100ns(uint32_t number_of_100ns)
{

	number_of_100ns = number_of_100ns*8.4 -150;
	TIM2->CNT = 0x00;
	while(TIM2->CNT < number_of_100ns) // exact would be 16.8
		asm("\t nop");
}

void timer_2_Init(void){
	RCC->AHB1ENR |= 0x01; // 1: IO port A clock enabled

	RCC->APB1ENR  |= 0x01; // 1: enable TIM2

	//TIM2->ARR = 0xFFFFFF; // Set the timer reset on the highest possible value

	TIM2->CR1  |= 0x0001; //0001 Enable Timer
}

void trigger_awg_2g(void){
	GPIOA->ODR |= GPIO_PIN_1;
	GPIOA->ODR = 0x00;
}


void orange_input_pin_init(void){

	GPIO_InitTypeDef GPIO_InitStruct_2 = {0};
	GPIO_InitStruct_2.Pin = GPIO_PIN_2;
	GPIO_InitStruct_2.Mode = GPIO_MODE_INPUT;
	GPIO_InitStruct_2.Pull = GPIO_PULLDOWN;
	GPIO_InitStruct_2.Speed = GPIO_SPEED_FREQ_HIGH; // Maybe GPIO_SPEED_FAST would be enough.
	HAL_GPIO_Init(GPIOA, &GPIO_InitStruct_2);
}


void trigger_awg_Init(void){
	/* GPIOA Ports Clock Enable */
	__HAL_RCC_GPIOA_CLK_ENABLE();

	GPIO_InitTypeDef GPIO_InitStruct = {0};
	GPIO_InitStruct.Pin = GPIO_PIN_1;
	GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
	GPIO_InitStruct.Pull = GPIO_NOPULL;
	GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH; // Maybe GPIO_SPEED_FAST would be enough.
	HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);
}
/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */

  /* USER CODE END Error_Handler_Debug */
}

#ifdef  USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{ 
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     tex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
